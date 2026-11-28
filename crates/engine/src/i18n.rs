//! Localisation: string tables per locale, the player's language choice, localised asset paths.
//!
//! * **Locale choice**: `ctx.i18n.locale()` / `set_locale("de")`. Startup order: page URL
//!   `?lang=de` (testing, not stored) > the stored choice (localStorage `wasmonly.locale`, see
//!   [`I18n::set_storage_key`]) > the default locale (`"en"`, [`I18n::set_default`]). A switch is
//!   stored, sets `<html lang>` and raises [`I18n::changed`] for the next frame (rebuild HUD /
//!   menus / dialog then; nothing reloads). [`I18n::revision`] counts switches + table loads.
//! * **Tables**: JSON per locale, nested objects are flattened with `.` (`{"menu": {"play": ".."}}`
//!   = key `menu.play`). An object whose keys are all CLDR plural categories (`zero`, `one`, `two`,
//!   `few`, `many`, `other`) and that has `other` is a plural entry. Keys starting with `@` are
//!   meta data: `"@name": "Deutsch"` = the locale's display name. `add_locale(code, include_str!(..))`
//!   or `ctx.load_locale(code, url)` (fetched; raises `changed` when it is the current locale).
//! * **Lookup**: `t(key)`: current locale, its language without region (`de-AT` -> `de`), the
//!   default locale, then the key itself. Placeholders `{name}` are filled by `tf(key, &[..])` or
//!   the [`t!`](crate::t) macro (`t!(ctx.i18n, "coins", n = 3)`); `{{` / `}}` are literal braces;
//!   an unknown placeholder stays as written. A plural entry picks its form by the argument `n`
//!   (or [`I18n::tn`]) with the current locale's plural rule. Keys that fell back or were missing
//!   are recorded (`locale` API: `fell_back`, `unknown`).
//! * **Assets**: `path("assets/voice/{locale}/x.mp3")` = the current locale's file, `paths(..)` the
//!   candidates in lookup order; `ctx.load_sound_localized(template)` loads the first that exists
//!   (falls back to the default locale's file on HTTP errors).
//! * **Checks**: [`I18n::report`] = keys missing in a locale (vs the default), extra keys,
//!   placeholder / plural mismatches. AI API `locale {missing: "de"}`.

use serde_json::{json, Map, Value};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Display;

/// localStorage key of the player's choice (default).
pub const DEFAULT_STORAGE_KEY: &str = "wasmonly.locale";

const PLURAL_CATS: [&str; 6] = ["zero", "one", "two", "few", "many", "other"];

#[derive(Clone, Debug)]
pub enum Entry {
    Text(String),
    /// (category, text); always contains `other`.
    Plural(Vec<(String, String)>),
}

impl Entry {
    fn texts(&self) -> Vec<&str> {
        match self {
            Entry::Text(s) => vec![s.as_str()],
            Entry::Plural(f) => f.iter().map(|(_, s)| s.as_str()).collect(),
        }
    }
}

#[derive(Default, Debug)]
pub struct Table {
    pub name: String,
    pub entries: HashMap<String, Entry>,
}

/// Where the current locale came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocaleSource {
    Default,
    Stored,
    Url,
    Set,
}

impl LocaleSource {
    fn as_str(self) -> &'static str {
        match self {
            LocaleSource::Default => "default",
            LocaleSource::Stored => "stored",
            LocaleSource::Url => "url",
            LocaleSource::Set => "set",
        }
    }
}

pub struct I18n {
    tables: BTreeMap<String, Table>,
    current: String,
    default: String,
    source: LocaleSource,
    storage_key: String,
    changed: bool,
    revision: u64,
    /// key -> locale it was found in (looked up but not in the current locale).
    fell_back: RefCell<BTreeMap<String, String>>,
    /// Keys found in no table.
    unknown: RefCell<BTreeSet<String>>,
}

impl Default for I18n {
    fn default() -> Self {
        let mut s = Self {
            tables: BTreeMap::new(),
            current: "en".into(),
            default: "en".into(),
            source: LocaleSource::Default,
            storage_key: DEFAULT_STORAGE_KEY.into(),
            changed: false,
            revision: 0,
            fell_back: Default::default(),
            unknown: Default::default(),
        };
        s.pick_startup();
        s
    }
}

fn storage() -> Option<web_sys::Storage> {
    web_sys::window()?.local_storage().ok()?
}

fn url_lang() -> Option<String> {
    let loc = js_sys::Reflect::get(&js_sys::global(), &"location".into()).ok()?;
    let search = js_sys::Reflect::get(&loc, &"search".into()).ok()?.as_string()?;
    search.trim_start_matches('?').split('&').find_map(|kv| kv.strip_prefix("lang=")).map(normalize).filter(|s| !s.is_empty())
}

/// `de_AT` / `DE-at` -> `de-at`.
pub fn normalize(code: &str) -> String {
    code.trim().replace('_', "-").to_ascii_lowercase()
}

/// CLDR plural category of `n` for a locale (cardinal; the common rules, `other` for languages
/// without plural forms).
pub fn plural_category(locale: &str, n: f64) -> &'static str {
    let lang = locale.split('-').next().unwrap_or(locale);
    let int = n.fract() == 0.0;
    let i = n.abs().trunc() as u64;
    match lang {
        "ja" | "zh" | "ko" | "th" | "vi" | "id" | "ms" | "tr" => "other",
        "fr" | "pt" => if n.abs() < 2.0 { "one" } else { "other" },
        "ru" | "uk" | "be" | "sr" | "hr" | "bs" => {
            if !int {
                "other"
            } else if i % 10 == 1 && i % 100 != 11 {
                "one"
            } else if (2..=4).contains(&(i % 10)) && !(12..=14).contains(&(i % 100)) {
                "few"
            } else {
                "many"
            }
        }
        "pl" => {
            if !int {
                "other"
            } else if i == 1 {
                "one"
            } else if (2..=4).contains(&(i % 10)) && !(12..=14).contains(&(i % 100)) {
                "few"
            } else {
                "many"
            }
        }
        "cs" | "sk" => if !int { "many" } else if i == 1 { "one" } else if (2..=4).contains(&i) { "few" } else { "other" },
        _ => if int && i == 1 { "one" } else { "other" },
    }
}

/// Placeholder names in a text (`{name}`, not `{{`).
fn placeholders(s: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'{' {
            if b.get(i + 1) == Some(&b'{') {
                i += 2;
                continue;
            }
            if let Some(end) = s[i + 1..].find('}') {
                let name = &s[i + 1..i + 1 + end];
                if !name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '_') {
                    out.insert(name.to_string());
                }
                i += end + 2;
                continue;
            }
        }
        i += 1;
    }
    out
}

/// Fills `{name}` placeholders; `{{` / `}}` = literal braces; unknown names stay as written.
pub fn format(s: &str, args: &[(&str, String)]) -> String {
    let mut out = String::with_capacity(s.len() + 16);
    let mut rest = s;
    while let Some(p) = rest.find(['{', '}']) {
        out.push_str(&rest[..p]);
        let tail = &rest[p..];
        if tail.starts_with("{{") || tail.starts_with("}}") {
            out.push_str(&tail[..1]);
            rest = &tail[2..];
            continue;
        }
        if tail.starts_with('{') {
            if let Some(end) = tail.find('}') {
                let name = &tail[1..end];
                if let Some((_, v)) = args.iter().find(|(k, _)| *k == name) {
                    out.push_str(v);
                    rest = &tail[end + 1..];
                    continue;
                }
            }
        }
        out.push_str(&tail[..1]);
        rest = &tail[1..];
    }
    out.push_str(rest);
    out
}

fn flatten(prefix: &str, v: &Value, table: &mut Table, errors: &mut Vec<String>) {
    let key = |k: &str| if prefix.is_empty() { k.to_string() } else { format!("{prefix}.{k}") };
    match v {
        Value::Object(m) => {
            let plural = m.contains_key("other") && m.keys().all(|k| PLURAL_CATS.contains(&k.as_str()));
            if plural && !prefix.is_empty() {
                let forms = m.iter().filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string()))).collect::<Vec<_>>();
                if forms.len() != m.len() {
                    errors.push(format!("{prefix}: plural forms must be strings"));
                }
                table.entries.insert(prefix.to_string(), Entry::Plural(forms));
                return;
            }
            for (k, v) in m {
                if prefix.is_empty() && k.starts_with('@') {
                    if k == "@name" {
                        if let Some(s) = v.as_str() {
                            table.name = s.to_string();
                        }
                    }
                    continue;
                }
                flatten(&key(k), v, table, errors);
            }
        }
        Value::String(s) => {
            table.entries.insert(prefix.to_string(), Entry::Text(s.clone()));
        }
        Value::Number(_) | Value::Bool(_) => {
            table.entries.insert(prefix.to_string(), Entry::Text(v.to_string()));
        }
        Value::Null => {}
        Value::Array(_) => errors.push(format!("{prefix}: arrays are not supported (use keys {prefix}.0, ..)")),
    }
}

impl I18n {
    fn pick_startup(&mut self) {
        if let Some(l) = url_lang() {
            self.current = l;
            self.source = LocaleSource::Url;
        } else if let Some(l) = storage().and_then(|s| s.get_item(&self.storage_key).ok().flatten()).map(|s| normalize(&s)).filter(|s| !s.is_empty()) {
            self.current = l;
            self.source = LocaleSource::Stored;
        } else {
            self.current = self.default.clone();
            self.source = LocaleSource::Default;
        }
        set_html_lang(&self.current);
    }

    /// The current locale code (lower case, `-` separated).
    pub fn locale(&self) -> &str {
        &self.current
    }

    pub fn default_locale(&self) -> &str {
        &self.default
    }

    pub fn source(&self) -> LocaleSource {
        self.source
    }

    /// The fallback locale (default "en"). Changes the current one too when no choice was made yet.
    pub fn set_default(&mut self, code: &str) {
        self.default = normalize(code);
        if self.source == LocaleSource::Default && self.current != self.default {
            self.current = self.default.clone();
            self.mark_changed();
            set_html_lang(&self.current);
        }
    }

    /// Own localStorage key for the choice (several games on one origin). Re-reads the stored
    /// choice under the new key (unless the URL picked the locale).
    pub fn set_storage_key(&mut self, key: &str) {
        self.storage_key = key.to_string();
        if self.source != LocaleSource::Url && self.source != LocaleSource::Set {
            let before = self.current.clone();
            self.pick_startup();
            if self.current != before {
                self.mark_changed();
            }
        }
    }

    /// Switches the locale, stores the choice and raises [`changed`](Self::changed). Unknown codes
    /// (no table under that code or its language) are an error.
    pub fn set_locale(&mut self, code: &str) -> Result<(), String> {
        let code = normalize(code);
        if !self.tables.is_empty() && self.resolve_table(&code).is_none() {
            return Err(format!("unknown locale '{code}' (available: {})", self.tables.keys().cloned().collect::<Vec<_>>().join(", ")));
        }
        if let Some(s) = storage() {
            let _ = s.set_item(&self.storage_key, &code);
        }
        self.source = LocaleSource::Set;
        if code != self.current {
            crate::log_info!("locale: {} -> {code}", self.current);
            self.current = code;
            set_html_lang(&self.current);
            self.fell_back.borrow_mut().clear();
            self.unknown.borrow_mut().clear();
            self.mark_changed();
        }
        Ok(())
    }

    /// Forgets the stored choice (the next start uses the default / URL).
    pub fn forget_choice(&mut self) {
        if let Some(s) = storage() {
            let _ = s.remove_item(&self.storage_key);
        }
    }

    /// True during the frame after a locale switch or after the current locale's table changed:
    /// rebuild text then. Cleared at the end of each frame.
    pub fn changed(&self) -> bool {
        self.changed
    }

    /// Counts switches and table loads (compare with a stored value for lazy rebuilds).
    pub fn revision(&self) -> u64 {
        self.revision
    }

    fn mark_changed(&mut self) {
        self.changed = true;
        self.revision += 1;
    }

    pub(crate) fn end_frame(&mut self) {
        self.changed = false;
    }

    /// Adds (or replaces) a locale's table from JSON. Returns the number of keys.
    pub fn add_locale(&mut self, code: &str, json: &str) -> Result<usize, String> {
        let v: Value = serde_json::from_str(json).map_err(|e| format!("locale {code}: {e}"))?;
        self.add_locale_value(code, &v)
    }

    pub fn add_locale_value(&mut self, code: &str, v: &Value) -> Result<usize, String> {
        let code = normalize(code);
        if !v.is_object() {
            return Err(format!("locale {code}: the table must be a JSON object"));
        }
        let mut table = Table::default();
        let mut errors = Vec::new();
        flatten("", v, &mut table, &mut errors);
        for e in &errors {
            crate::log_warn!("locale {code}: {e}");
        }
        if table.name.is_empty() {
            table.name = code.clone();
        }
        let n = table.entries.len();
        self.tables.insert(code.clone(), table);
        // a new table can change what the current locale shows
        if self.chain().iter().any(|c| *c == code) {
            self.fell_back.borrow_mut().clear();
            self.unknown.borrow_mut().clear();
            self.mark_changed();
        }
        Ok(n)
    }

    /// Codes with a table, sorted.
    pub fn available(&self) -> Vec<&str> {
        self.tables.keys().map(|s| s.as_str()).collect()
    }

    /// Display name of a locale (`@name` in its table, else the code).
    pub fn name(&self, code: &str) -> Option<&str> {
        self.tables.get(&normalize(code)).map(|t| t.name.as_str())
    }

    fn resolve_table(&self, code: &str) -> Option<&str> {
        if let Some((k, _)) = self.tables.get_key_value(code) {
            return Some(k);
        }
        let lang = code.split('-').next()?;
        self.tables.get_key_value(lang).map(|(k, _)| k.as_str())
    }

    /// Locales searched for a key, in order: current, its language, default.
    fn chain(&self) -> Vec<&str> {
        let mut v: Vec<&str> = vec![self.current.as_str()];
        if let Some(lang) = self.current.split('-').next() {
            if lang != self.current {
                v.push(lang);
            }
        }
        if !v.contains(&self.default.as_str()) {
            v.push(self.default.as_str());
        }
        v
    }

    fn lookup(&self, key: &str) -> Option<(&str, &Entry)> {
        let chain = self.chain();
        for (i, code) in chain.iter().enumerate() {
            if let Some(e) = self.tables.get(*code).and_then(|t| t.entries.get(key)) {
                if i > 0 && !self.tables.is_empty() {
                    let mut fb = self.fell_back.borrow_mut();
                    if !fb.contains_key(key) {
                        fb.insert(key.to_string(), code.to_string());
                    }
                }
                return Some((code, e));
            }
        }
        if !self.tables.is_empty() {
            self.unknown.borrow_mut().insert(key.to_string());
        }
        None
    }

    /// Whether any locale in the lookup chain has the key.
    pub fn has(&self, key: &str) -> bool {
        self.chain().iter().any(|c| self.tables.get(*c).is_some_and(|t| t.entries.contains_key(key)))
    }

    /// Text of `key` (no placeholders filled; a plural entry gives its `other` form).
    pub fn t(&self, key: &str) -> String {
        self.tf(key, &[])
    }

    /// Text of `key` with placeholders filled; a plural entry picks its form by the argument `n`.
    pub fn tf(&self, key: &str, args: &[(&str, &dyn Display)]) -> String {
        let args: Vec<(&str, String)> = args.iter().map(|(k, v)| (*k, v.to_string())).collect();
        let n = args.iter().find(|(k, _)| *k == "n").and_then(|(_, v)| v.parse::<f64>().ok());
        self.render(key, n, &args)
    }

    /// Plural lookup with an explicit count (`{n}` is filled with it too).
    pub fn tn(&self, key: &str, n: f64, args: &[(&str, &dyn Display)]) -> String {
        let mut a: Vec<(&str, String)> = args.iter().map(|(k, v)| (*k, v.to_string())).collect();
        if !a.iter().any(|(k, _)| *k == "n") {
            a.push(("n", n.to_string()));
        }
        self.render(key, Some(n), &a)
    }

    fn render(&self, key: &str, n: Option<f64>, args: &[(&str, String)]) -> String {
        let Some((code, e)) = self.lookup(key) else { return key.to_string() };
        let text = match e {
            Entry::Text(s) => s.as_str(),
            Entry::Plural(forms) => {
                let get = |c: &str| forms.iter().find(|(k, _)| k == c).map(|(_, s)| s.as_str());
                let mut pick = None;
                if let Some(n) = n {
                    if n == 0.0 {
                        pick = get("zero");
                    }
                    if pick.is_none() {
                        pick = get(plural_category(code, n));
                    }
                }
                pick.or_else(|| get("other")).unwrap_or("")
            }
        };
        if args.is_empty() && !text.contains(['{', '}']) {
            return text.to_string();
        }
        format(text, args)
    }

    /// `{locale}` in a path replaced by the current locale (no existence check).
    pub fn path(&self, template: &str) -> String {
        template.replace("{locale}", &self.current)
    }

    /// Candidate paths in lookup order (current, its language, default), duplicates removed.
    pub fn paths(&self, template: &str) -> Vec<String> {
        let mut v: Vec<String> = Vec::new();
        for c in self.chain() {
            let p = template.replace("{locale}", c);
            if !v.contains(&p) {
                v.push(p);
            }
        }
        v
    }

    /// Translation check of `code` against the default locale: missing keys, extra keys, keys whose
    /// placeholders differ, plural entries in one and plain text in the other.
    pub fn report(&self, code: &str) -> Value {
        let code = normalize(code);
        let (Some(t), Some(base)) = (self.tables.get(&code), self.tables.get(&self.default)) else {
            return json!({"error": format!("no table for '{code}' or the default '{}'", self.default)});
        };
        let mut missing: Vec<&str> = base.entries.keys().filter(|k| !t.entries.contains_key(*k)).map(|s| s.as_str()).collect();
        let mut extra: Vec<&str> = t.entries.keys().filter(|k| !base.entries.contains_key(*k)).map(|s| s.as_str()).collect();
        missing.sort();
        extra.sort();
        let mut placeholder = Vec::new();
        let mut kind = Vec::new();
        let mut keys: Vec<&String> = t.entries.keys().collect();
        keys.sort();
        for k in keys {
            let Some(b) = base.entries.get(k) else { continue };
            let e = &t.entries[k];
            if matches!(e, Entry::Plural(_)) != matches!(b, Entry::Plural(_)) {
                kind.push(k.as_str());
            }
            let ph = |e: &Entry| e.texts().iter().flat_map(|s| placeholders(s)).filter(|p| p != "n").collect::<BTreeSet<_>>();
            let (a, bb) = (ph(e), ph(b));
            if a != bb {
                placeholder.push(json!({"key": k, "has": a, "default_has": bb}));
            }
        }
        json!({
            "locale": code, "default": self.default, "keys": t.entries.len(), "default_keys": base.entries.len(),
            "missing": missing, "extra": extra, "placeholder_mismatch": placeholder, "plural_mismatch": kind,
            "complete": missing.is_empty() && placeholder.is_empty(),
        })
    }

    /// State for the `locale` command.
    pub fn info(&self) -> Value {
        let locales: Vec<Value> = self.tables.iter().map(|(c, t)| json!({"code": c, "name": t.name, "keys": t.entries.len()})).collect();
        json!({
            "current": self.current, "default": self.default, "source": self.source.as_str(),
            "has_table": self.resolve_table(&self.current).is_some(),
            "storage_key": self.storage_key,
            "stored": storage().and_then(|s| s.get_item(&self.storage_key).ok().flatten()),
            "revision": self.revision, "changed": self.changed, "locales": locales,
            "fell_back_count": self.fell_back.borrow().len(),
            "fell_back": self.fell_back.borrow().iter().map(|(k, c)| (k.clone(), Value::String(c.clone()))).collect::<Map<_, _>>(),
            "unknown": self.unknown.borrow().iter().cloned().collect::<Vec<_>>(),
        })
    }
}

fn set_html_lang(code: &str) {
    if let Some(el) = web_sys::window().and_then(|w| w.document()).and_then(|d| d.document_element()) {
        let _ = el.set_attribute("lang", code);
    }
}

/// `t!(ctx.i18n, "key")`, `t!(ctx.i18n, "greet", name = hero, n = 3)`: lookup with placeholders
/// (`n` also picks the plural form).
#[macro_export]
macro_rules! t {
    ($i18n:expr, $key:expr) => {
        $i18n.t($key)
    };
    ($i18n:expr, $key:expr, $($name:ident = $val:expr),+ $(,)?) => {
        $i18n.tf($key, &[$((stringify!($name), &$val as &dyn ::std::fmt::Display)),+])
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fmt() {
        assert_eq!(format("Hi {name}! {{x}} {missing}", &[("name", "Link".into())]), "Hi Link! {x} {missing}");
        assert_eq!(plural_category("de", 1.0), "one");
        assert_eq!(plural_category("ru", 22.0), "few");
        assert_eq!(placeholders("{a} {{b}} {c}"), ["a", "c"].iter().map(|s| s.to_string()).collect());
    }
}

impl I18n {
    /// AI API `locale`.
    pub fn command(&mut self, a: &Value) -> Result<Value, String> {
        if let (Some(code), Some(table)) = (a.get("add").and_then(|v| v.as_str()), a.get("table")) {
            let n = self.add_locale_value(code, table)?;
            return Ok(json!({"added": normalize(code), "keys": n, "info": self.info()}));
        }
        if let Some(code) = a.get("set").and_then(|v| v.as_str()) {
            self.set_locale(code)?;
            return Ok(self.info());
        }
        if a.get("forget").and_then(|v| v.as_bool()) == Some(true) {
            self.forget_choice();
            return Ok(self.info());
        }
        if let Some(m) = a.get("missing") {
            return Ok(match m.as_str() {
                Some(code) => self.report(code),
                None => Value::Array(self.available().iter().filter(|c| **c != self.default).map(|c| self.report(c)).collect()),
            });
        }
        if let Some(key) = a.get("t").and_then(|v| v.as_str()) {
            let args: Vec<(String, String)> = a
                .get("args")
                .and_then(|v| v.as_object())
                .map(|m| m.iter().map(|(k, v)| (k.clone(), v.as_str().map(|s| s.to_string()).unwrap_or_else(|| v.to_string()))).collect())
                .unwrap_or_default();
            let dargs: Vec<(&str, &dyn Display)> = args.iter().map(|(k, v)| (k.as_str(), v as &dyn Display)).collect();
            let text = match a.get("n").and_then(|v| v.as_f64()) {
                Some(n) => self.tn(key, n, &dargs),
                None => self.tf(key, &dargs),
            };
            return Ok(json!({"key": key, "locale": self.current, "text": text, "found": self.has(key)}));
        }
        Ok(self.info())
    }
}
