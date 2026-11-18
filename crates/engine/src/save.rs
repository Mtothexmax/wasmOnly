//! Save games and persistent data.
//!
//! Two stores, one per job:
//! * **Slots** (`localStorage`, synchronous): small JSON state such as progress, settings and
//!   unlocks. Each slot is an envelope `{"v": version, "t": unix_ms, "data": ...}` under the key
//!   `<namespace>:<slot>`. Reads return the stored version, so a game can migrate old saves.
//! * **Blobs** (IndexedDB, asynchronous): large binary data such as screenshots, replays and
//!   generated assets. Database `<namespace>`, object store `blobs`, value = `Uint8Array`.
//!
//! ```ignore
//! ctx.save.set_namespace("tumble", 2);              // once in init: name + current save version
//! ctx.save.write("slot1", &progress)?;               // any Serialize
//! if let Some(p) = ctx.save.read::<Progress>("slot1") { ... }
//! let e = ctx.save.read_entry("slot1");              // {version, saved_at, data: Value} for migrations
//! wasm_bindgen_futures::spawn_local(async move {
//!     engine::save::put_blob("tumble", "shot1", &png).await.ok();
//!     let bytes = engine::save::get_blob("tumble", "shot1").await;
//! });
//! ```
//!
//! AI API: `save` lists/reads/writes/deletes slots; `save {op:"blobs"}` lists IndexedDB keys.

use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::{json, Value};
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{IdbDatabase, IdbRequest, IdbTransactionMode, Storage};

const BLOB_STORE: &str = "blobs";

/// A slot as stored: save-format version, save time (unix ms) and the game's data.
#[derive(Clone, Debug, Serialize)]
pub struct SaveEntry {
    pub version: u32,
    pub saved_at: f64,
    pub data: Value,
}

pub struct Save {
    namespace: String,
    version: u32,
    /// IndexedDB keys + sizes, refreshed after every blob write/delete (for `save {op:"blobs"}`).
    pub(crate) blob_index: Rc<RefCell<Vec<(String, u32)>>>,
}

impl Default for Save {
    fn default() -> Self {
        let s = Self { namespace: "wasmonly".into(), version: 1, blob_index: Default::default() };
        s.refresh_blob_index();
        s
    }
}

fn storage() -> Result<Storage, String> {
    web_sys::window()
        .ok_or("no window")?
        .local_storage()
        .map_err(|e| format!("localStorage: {e:?}"))?
        .ok_or_else(|| "localStorage unavailable".to_string())
}

impl Save {
    /// Namespace (key prefix / database name) and the current save-format version.
    pub fn set_namespace(&mut self, namespace: &str, version: u32) {
        self.namespace = namespace.to_string();
        self.version = version;
        self.refresh_blob_index();
    }

    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    fn key(&self, slot: &str) -> String {
        format!("{}:{slot}", self.namespace)
    }

    /// Writes `data` into `slot` with the current version.
    pub fn write<T: Serialize>(&self, slot: &str, data: &T) -> Result<(), String> {
        let data = serde_json::to_value(data).map_err(|e| e.to_string())?;
        self.write_value(slot, data)
    }

    pub fn write_value(&self, slot: &str, data: Value) -> Result<(), String> {
        let env = json!({"v": self.version, "t": js_sys::Date::now(), "data": data});
        storage()?
            .set_item(&self.key(slot), &env.to_string())
            .map_err(|e| format!("save {slot}: {e:?} (quota?)"))?;
        crate::log_info!("save: wrote slot '{slot}' (v{})", self.version);
        Ok(())
    }

    /// The slot with its version and time, or None if empty/corrupt (corrupt slots log a warning).
    pub fn read_entry(&self, slot: &str) -> Option<SaveEntry> {
        let raw = storage().ok()?.get_item(&self.key(slot)).ok()??;
        match serde_json::from_str::<Value>(&raw) {
            Ok(v) if v.get("data").is_some() => Some(SaveEntry {
                version: v["v"].as_u64().unwrap_or(0) as u32,
                saved_at: v["t"].as_f64().unwrap_or(0.0),
                data: v["data"].clone(),
            }),
            // raw JSON written by something else: version 0
            Ok(v) => Some(SaveEntry { version: 0, saved_at: 0.0, data: v }),
            Err(e) => {
                crate::log_warn!("save: slot '{slot}' is corrupt: {e}");
                None
            }
        }
    }

    /// Deserializes the slot's data. Logs a warning on version mismatch or schema errors;
    /// use [`Self::read_entry`] to migrate old versions yourself.
    pub fn read<T: DeserializeOwned>(&self, slot: &str) -> Option<T> {
        let e = self.read_entry(slot)?;
        if e.version != self.version {
            crate::log_warn!("save: slot '{slot}' has version {} (current {})", e.version, self.version);
        }
        match serde_json::from_value(e.data) {
            Ok(t) => Some(t),
            Err(err) => {
                crate::log_warn!("save: slot '{slot}' does not match the current format: {err}");
                None
            }
        }
    }

    pub fn exists(&self, slot: &str) -> bool {
        storage().ok().and_then(|s| s.get_item(&self.key(slot)).ok().flatten()).is_some()
    }

    pub fn delete(&self, slot: &str) {
        if let Ok(s) = storage() {
            let _ = s.remove_item(&self.key(slot));
        }
    }

    /// Slots of this namespace: (slot, version, saved_at, bytes).
    pub fn list(&self) -> Vec<Value> {
        let Ok(s) = storage() else { return vec![] };
        let prefix = format!("{}:", self.namespace);
        let mut out = vec![];
        for i in 0..s.length().unwrap_or(0) {
            let Some(k) = s.key(i).ok().flatten() else { continue };
            let Some(slot) = k.strip_prefix(&prefix) else { continue };
            let raw = s.get_item(&k).ok().flatten().unwrap_or_default();
            let v: Value = serde_json::from_str(&raw).unwrap_or(Value::Null);
            out.push(json!({"slot": slot, "version": v["v"], "saved_at": v["t"], "bytes": raw.len()}));
        }
        out.sort_by(|a, b| a["slot"].as_str().cmp(&b["slot"].as_str()));
        out
    }

    /// Re-reads the IndexedDB key list in the background.
    pub fn refresh_blob_index(&self) {
        let (ns, index) = (self.namespace.clone(), self.blob_index.clone());
        wasm_bindgen_futures::spawn_local(async move {
            match list_blobs(&ns).await {
                Ok(v) => *index.borrow_mut() = v,
                Err(e) => crate::log_warn!("save: blob index: {e}"),
            }
        });
    }

    /// Stores `bytes` under `key` in IndexedDB in the background (errors are logged).
    pub fn put_blob_bg(&self, key: &str, bytes: Vec<u8>) {
        let (ns, key, index) = (self.namespace.clone(), key.to_string(), self.blob_index.clone());
        wasm_bindgen_futures::spawn_local(async move {
            match put_blob(&ns, &key, &bytes).await {
                Ok(()) => {
                    if let Ok(v) = list_blobs(&ns).await {
                        *index.borrow_mut() = v;
                    }
                }
                Err(e) => crate::log_error!("save: put_blob {key}: {e}"),
            }
        });
    }

    /// AI API `save`.
    pub(crate) fn command(&self, a: &Value) -> Result<Value, String> {
        let slot = a["slot"].as_str();
        match a["op"].as_str().unwrap_or(if slot.is_some() { "get" } else { "list" }) {
            "list" => Ok(json!({"namespace": self.namespace, "version": self.version, "slots": self.list(),
                "blobs": self.blob_index.borrow().len()})),
            "get" => {
                let slot = slot.ok_or("slot required")?;
                Ok(serde_json::to_value(self.read_entry(slot)).unwrap_or(Value::Null))
            }
            "set" => {
                let slot = slot.ok_or("slot required")?;
                self.write_value(slot, a["data"].clone())?;
                Ok(json!({"ok": true}))
            }
            "delete" => {
                let slot = slot.ok_or("slot required")?;
                self.delete(slot);
                Ok(json!({"ok": true}))
            }
            "blobs" => {
                self.refresh_blob_index();
                let b: Vec<Value> = self.blob_index.borrow().iter().map(|(k, n)| json!({"key": k, "bytes": n})).collect();
                Ok(json!({"blobs": b, "note": "index is refreshed asynchronously; call again for the newest state"}))
            }
            "delete_blob" => {
                let key = a["key"].as_str().ok_or("key required")?.to_string();
                let (ns, index) = (self.namespace.clone(), self.blob_index.clone());
                wasm_bindgen_futures::spawn_local(async move {
                    if let Err(e) = delete_blob(&ns, &key).await {
                        crate::log_warn!("save: delete_blob {key}: {e}");
                    }
                    if let Ok(v) = list_blobs(&ns).await {
                        *index.borrow_mut() = v;
                    }
                });
                Ok(json!({"ok": true}))
            }
            o => Err(format!("unknown op '{o}' (list|get|set|delete|blobs|delete_blob)")),
        }
    }
}

// ---------- IndexedDB (async) ----------

fn js(e: JsValue) -> String {
    format!("{e:?}")
}

/// Resolves when an IDB request succeeds (with its result) or rejects on error.
async fn wait(req: &IdbRequest) -> Result<JsValue, String> {
    let r = req.clone();
    let p = js_sys::Promise::new(&mut |resolve, reject| {
        let r2 = r.clone();
        let ok = Closure::once_into_js(move |_: web_sys::Event| {
            let _ = resolve.call1(&JsValue::NULL, &r2.result().unwrap_or(JsValue::NULL));
        });
        let r3 = r.clone();
        let err = Closure::once_into_js(move |_: web_sys::Event| {
            let msg = r3.error().ok().flatten().map(|e| e.message()).unwrap_or_else(|| "IndexedDB error".into());
            let _ = reject.call1(&JsValue::NULL, &msg.into());
        });
        r.set_onsuccess(Some(ok.unchecked_ref()));
        r.set_onerror(Some(err.unchecked_ref()));
    });
    wasm_bindgen_futures::JsFuture::from(p).await.map_err(js)
}

async fn open(namespace: &str) -> Result<IdbDatabase, String> {
    let factory = web_sys::window().ok_or("no window")?.indexed_db().map_err(js)?.ok_or("IndexedDB unavailable")?;
    let req = factory.open_with_u32(namespace, 1).map_err(js)?;
    let up = req.clone();
    let on_upgrade = Closure::once_into_js(move |_: web_sys::Event| {
        if let Ok(db) = up.result().and_then(|r| r.dyn_into::<IdbDatabase>()) {
            if !db.object_store_names().contains(BLOB_STORE) {
                let _ = db.create_object_store(BLOB_STORE);
            }
        }
    });
    req.set_onupgradeneeded(Some(on_upgrade.unchecked_ref()));
    let db = wait(&req).await?;
    db.dyn_into::<IdbDatabase>().map_err(js)
}

async fn store(namespace: &str, mode: IdbTransactionMode) -> Result<web_sys::IdbObjectStore, String> {
    let db = open(namespace).await?;
    let tx = db.transaction_with_str_and_mode(BLOB_STORE, mode).map_err(js)?;
    tx.object_store(BLOB_STORE).map_err(js)
}

/// Stores `bytes` under `key` (database = namespace).
pub async fn put_blob(namespace: &str, key: &str, bytes: &[u8]) -> Result<(), String> {
    let s = store(namespace, IdbTransactionMode::Readwrite).await?;
    let arr = js_sys::Uint8Array::from(bytes);
    wait(&s.put_with_key(&arr, &key.into()).map_err(js)?).await.map(|_| ())
}

/// Bytes under `key`, `None` if absent.
pub async fn get_blob(namespace: &str, key: &str) -> Result<Option<Vec<u8>>, String> {
    let s = store(namespace, IdbTransactionMode::Readonly).await?;
    let v = wait(&s.get(&key.into()).map_err(js)?).await?;
    if v.is_undefined() || v.is_null() {
        return Ok(None);
    }
    Ok(Some(js_sys::Uint8Array::new(&v).to_vec()))
}

pub async fn delete_blob(namespace: &str, key: &str) -> Result<(), String> {
    let s = store(namespace, IdbTransactionMode::Readwrite).await?;
    wait(&s.delete(&key.into()).map_err(js)?).await.map(|_| ())
}

/// (key, bytes) of every blob.
pub async fn list_blobs(namespace: &str) -> Result<Vec<(String, u32)>, String> {
    let s = store(namespace, IdbTransactionMode::Readonly).await?;
    let keys = wait(&s.get_all_keys().map_err(js)?).await?;
    let vals = wait(&s.get_all().map_err(js)?).await?;
    let (keys, vals) = (js_sys::Array::from(&keys), js_sys::Array::from(&vals));
    Ok((0..keys.length())
        .map(|i| {
            let k = keys.get(i).as_string().unwrap_or_default();
            let n = js_sys::Uint8Array::new(&vals.get(i)).length();
            (k, n)
        })
        .collect())
}
