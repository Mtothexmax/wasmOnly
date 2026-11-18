//! Sky, atmosphere and the day/night cycle.
//!
//! * `render.sky_mode`: 0 = gradient (`sky_zenith`/`sky_horizon`), 1 = physically based atmosphere
//!   (Rayleigh + Mie + ozone single scattering, rendered each frame into a small sky-view LUT by
//!   `shaders/sky_lut.wgsl`, sampled by the sky, fog, clouds and water reflections).
//! * Procedural clouds (`clouds` = coverage), stars and a moon with phases, an optional
//!   equirectangular skybox image ([`Ctx::load_skybox`], blended by `skybox_blend` by day and
//!   `skybox_night_blend` by night).
//! * Day/night cycle (`day_cycle`): `time_of_day` (hours) places sun and moon on their arcs
//!   (`sun_azimuth_deg` = sunrise direction, `sun_max_elevation_deg` = noon height) and drives the
//!   light direction/colour/intensity (sun by day, moon by night), ambient sky + ground colours,
//!   the gradient colours and an exposure boost at night. It advances with the simulation clock
//!   (`day_length_minutes` per 24 h; 0 = time stands still), so pausing stops it and tests stay
//!   deterministic. Games read [`SkyState`] (`ctx.renderer.sky`): `daylight`, `night`, sun/moon.
//!
//! AI API: `sky` (state), `sky {time, cycle, day_length, mode, clouds, stars, skybox_blend, ...}`.

use crate::renderer::RenderSettings;
use glam::Vec3;
use serde::Serialize;

/// Derived sky state, updated once per frame by [`update`].
#[derive(Clone, Debug, Default, Serialize)]
pub struct SkyState {
    /// Direction to the sun / moon (unit vectors, world space).
    pub sun: Vec3,
    pub moon: Vec3,
    /// 1 = full day light, 0 = sun below the horizon.
    pub daylight: f32,
    /// 0 = day, 1 = full night (stars, exposure boost; street lamps can follow it).
    pub night: f32,
    /// Sunlight after the atmosphere, times the daylight intensity (sun disk, clouds).
    pub sun_radiance: Vec3,
    /// Moon brightness used by the sky (disk, moonlit atmosphere, clouds).
    pub moon_brightness: f32,
    /// Which body lights the scene right now: "sun" or "moon".
    pub light: &'static str,
    /// Exposure multiplier applied on top of `render.exposure`.
    pub exposure_mul: f32,
    #[serde(skip)]
    last_sim: Option<f64>,
}

const R_GROUND: f32 = 6360e3;
const R_TOP: f32 = 6420e3;
const BETA_R: Vec3 = Vec3::new(5.8e-6, 13.5e-6, 33.1e-6);
const BETA_OZONE: Vec3 = Vec3::new(1.713e-6, 4.149e-6, 0.178e-6);
const BETA_M: f32 = 8e-6;
const H_R: f32 = 8000.0;
const H_M: f32 = 1200.0;

fn sphere(o: Vec3, d: Vec3, r: f32) -> (f32, f32) {
    let b = o.dot(d);
    let c = o.dot(o) - r * r;
    let disc = b * b - c;
    if disc < 0.0 {
        return (-1.0, -1.0);
    }
    let s = disc.sqrt();
    (-b - s, -b + s)
}

/// Optical depths (Rayleigh, Mie) from `p` along `d` to the top of the atmosphere.
fn optical_depth(p: Vec3, d: Vec3, steps: u32) -> (f32, f32) {
    let t = sphere(p, d, R_TOP).1.max(0.0);
    let dl = t / steps as f32;
    let (mut r, mut m) = (0.0, 0.0);
    for j in 0..steps {
        let h = ((p + d * (dl * (j as f32 + 0.5))).length() - R_GROUND).max(0.0);
        r += (-h / H_R).exp() * dl;
        m += (-h / H_M).exp() * dl;
    }
    (r, m)
}

fn extinction(r: f32, m: f32) -> Vec3 {
    let tau = (BETA_R + BETA_OZONE) * r + Vec3::splat(BETA_M * 1.1 * m);
    Vec3::new((-tau.x).exp(), (-tau.y).exp(), (-tau.z).exp())
}

/// Fraction of sunlight reaching the ground from direction `l` (to the sun). Same model as the LUT.
pub fn transmittance(l: Vec3) -> Vec3 {
    let o = Vec3::new(0.0, R_GROUND + 2.0, 0.0);
    if sphere(o, l, R_GROUND).0 > 0.0 {
        return Vec3::ZERO;
    }
    let (r, m) = optical_depth(o, l, 12);
    extinction(r, m)
}

/// Sky radiance in direction `d` for light from `l` (unit irradiance); CPU twin of `sky_lut.wgsl`.
pub fn scatter(d: Vec3, l: Vec3) -> Vec3 {
    let o = Vec3::new(0.0, R_GROUND + 2.0, 0.0);
    let mut t_max = sphere(o, d, R_TOP).1;
    let ground = sphere(o, d, R_GROUND);
    if ground.0 > 0.0 {
        t_max = ground.0;
    }
    let ds = t_max / 16.0;
    let (mut od_r, mut od_m) = (0.0, 0.0);
    let (mut sum_r, mut sum_m, mut sum_ms) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
    for i in 0..16 {
        let p = o + d * (ds * (i as f32 + 0.5));
        let h = p.length() - R_GROUND;
        let hr = (-h / H_R).exp() * ds;
        let hm = (-h / H_M).exp() * ds;
        od_r += hr;
        od_m += hm;
        if sphere(p, l, R_GROUND).0 > 0.0 {
            continue;
        }
        let (lr, lm) = optical_depth(p, l, 6);
        let att = extinction(od_r + lr, od_m + lm);
        sum_r += att * hr;
        sum_m += att * hm;
        sum_ms += att * hr * BETA_R * (BETA_R * 8000.0) * 0.64;
    }
    let mu = d.dot(l);
    let pi = std::f32::consts::PI;
    let phase_r = 3.0 / (16.0 * pi) * (1.0 + mu * mu);
    let g = 0.76f32;
    let g2 = g * g;
    let phase_m = 3.0 / (8.0 * pi) * ((1.0 - g2) * (1.0 + mu * mu)) / ((2.0 + g2) * (1.0 + g2 - 2.0 * g * mu).powf(1.5));
    sum_r * BETA_R * phase_r + sum_m * BETA_M * phase_m + sum_ms
}

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Point on the daily arc: `hours` 6 = rising in the `azimuth` direction, 12 = highest
/// (`max_elev`), 18 = setting opposite.
pub fn arc_dir(hours: f32, azimuth_deg: f32, max_elev_deg: f32) -> Vec3 {
    let az = azimuth_deg.to_radians();
    let east = Vec3::new(az.cos(), 0.0, az.sin());
    let side = Vec3::new(-az.sin(), 0.0, az.cos());
    let e = max_elev_deg.to_radians();
    let th = (hours - 6.0) / 24.0 * std::f32::consts::TAU;
    (east * th.cos() + (Vec3::Y * e.sin() + side * e.cos()) * th.sin()).normalize()
}

/// Advances the cycle by the simulation time since the last call and derives the lighting.
pub fn update(s: &mut RenderSettings, st: &mut SkyState, sim_time: f64) {
    let dt = st.last_sim.map(|l| (sim_time - l).clamp(0.0, 60.0)).unwrap_or(0.0) as f32;
    st.last_sim = Some(sim_time);
    if s.day_cycle && s.day_length_minutes > 0.0 {
        s.time_of_day = (s.time_of_day + dt * 24.0 / (s.day_length_minutes * 60.0)).rem_euclid(24.0);
    }
    st.exposure_mul = 1.0;
    if !s.day_cycle {
        // the sky follows the manually set light
        st.sun = (-s.sun_dir).normalize_or(Vec3::Y);
        st.moon = Vec3::ZERO;
        st.daylight = smoothstep(-0.04, 0.06, st.sun.y);
        st.night = 0.0;
        st.sun_radiance = s.sun_color * s.sun_intensity * transmittance(Vec3::new(st.sun.x, st.sun.y.max(0.02), st.sun.z)) / transmittance(Vec3::Y).max_element();
        st.moon_brightness = 0.0;
        st.light = "sun";
        return;
    }
    let sun = arc_dir(s.time_of_day, s.sun_azimuth_deg, s.sun_max_elevation_deg);
    let moon = arc_dir(s.time_of_day + 12.0 + 1.3, s.sun_azimuth_deg + 12.0, s.sun_max_elevation_deg * 0.85);
    st.sun = sun;
    st.moon = moon;
    let day = smoothstep(-0.04, 0.06, sun.y);
    st.daylight = day;
    st.night = 1.0 - smoothstep(-0.14, 0.02, sun.y);
    let noon_t = transmittance(Vec3::Y).max_element();
    let t_sun = transmittance(Vec3::new(sun.x, sun.y.max(0.015), sun.z).normalize());
    st.sun_radiance = t_sun * s.daylight_intensity / noon_t * smoothstep(-0.03, 0.02, sun.y);
    let moon_up = smoothstep(0.0, 0.12, moon.y);
    st.moon_brightness = s.moon_intensity * moon_up;

    // --- the one directional light: sun by day, moon by night (switch while both are ~dark)
    let moon_light = s.moon_intensity * moon_up * smoothstep(-0.04, -0.12, sun.y);
    if day > 0.0 || moon_light <= 0.0 {
        let d = Vec3::new(sun.x, sun.y.max(0.03), sun.z).normalize();
        s.sun_dir = -d;
        let m = t_sun.max_element().max(1e-4);
        s.sun_color = t_sun / m;
        s.sun_intensity = s.daylight_intensity * day * m / noon_t;
        st.light = "sun";
    } else {
        s.sun_dir = -Vec3::new(moon.x, moon.y.max(0.03), moon.z).normalize();
        s.sun_color = Vec3::new(0.62, 0.72, 1.0);
        s.sun_intensity = moon_light * 0.45;
        st.light = "moon";
    }

    // --- ambient + gradient colours from the same atmosphere
    let k = s.atmosphere_intensity;
    let sky_at = |d: Vec3| scatter(d.normalize(), sun) * k + scatter(d.normalize(), moon) * k * 0.004 * st.moon_brightness;
    let zenith = sky_at(Vec3::Y);
    let mut horizon = Vec3::ZERO;
    let mut ring = Vec3::ZERO;
    for i in 0..4 {
        let a = i as f32 * std::f32::consts::FRAC_PI_2 + 0.3;
        horizon += sky_at(Vec3::new(a.cos(), 0.05, a.sin())) / 4.0;
        ring += sky_at(Vec3::new(a.cos(), 0.6, a.sin())) / 4.0;
    }
    let airglow = Vec3::new(0.0012, 0.0016, 0.0032) * st.night;
    s.sky_zenith = zenith + airglow;
    s.sky_horizon = horizon + airglow;
    let night_amb = Vec3::new(0.010, 0.014, 0.028) * st.night * (0.4 + 0.6 * st.moon_brightness.min(1.0));
    s.sky_color = (zenith * 0.4 + ring * 0.4 + horizon * 0.2) * 0.95 + night_amb;
    s.ground_color = s.sun_color * s.sun_intensity * sun.y.max(0.0) * 0.035 + s.sky_color * 0.12;
    if s.day_cycle_exposure {
        st.exposure_mul = 1.0 + 1.6 * st.night;
    }
}

impl crate::app::Ctx {
    /// Loads an equirectangular sky image (PNG/JPG, 2:1) as the skybox. It shows once loaded,
    /// blended by `render.skybox_blend` (day) / `skybox_night_blend` (night).
    pub fn load_skybox(&mut self, url: &str) {
        let me = self.me.clone();
        let url = url.to_string();
        wasm_bindgen_futures::spawn_local(async move {
            let mime = if url.ends_with(".jpg") || url.ends_with(".jpeg") { "image/jpeg" } else { "image/png" };
            let bmp = match crate::gltf_loader::fetch_bytes(&url).await {
                Ok(b) => crate::gltf_loader::decode_image(&b, mime).await,
                Err(e) => Err(e),
            };
            let Some(rc) = me.upgrade() else { return };
            let Ok(mut ctx) = rc.try_borrow_mut() else {
                crate::log_error!("skybox {url}: engine busy when load finished");
                return;
            };
            match bmp.and_then(|b| ctx.renderer.add_texture_bitmap(&b, &format!("skybox:{url}"))) {
                Ok(id) => {
                    ctx.renderer.set_skybox(Some(id));
                    crate::log_info!("skybox loaded: {url}");
                }
                Err(e) => crate::log_error!("skybox {url}: {e}"),
            }
        });
    }

    /// AI API `sky`: state + quick controls (any other render setting via `render`).
    pub fn sky_command(&mut self, a: &serde_json::Value) -> Result<serde_json::Value, String> {
        use serde_json::json;
        let s = &mut self.renderer.settings;
        if let Some(t) = a.get("time").and_then(|v| v.as_f64()) {
            s.time_of_day = (t as f32).rem_euclid(24.0);
            s.day_cycle = true;
        }
        if let Some(v) = a.get("cycle").and_then(|v| v.as_bool()) {
            s.day_cycle = v;
        }
        if let Some(v) = a.get("day_length").and_then(|v| v.as_f64()) {
            s.day_length_minutes = v.max(0.0) as f32;
        }
        if let Some(v) = a.get("mode") {
            s.sky_mode = match v.as_str() {
                Some("gradient") => 0,
                Some("atmosphere") => 1,
                _ => v.as_u64().ok_or("mode: 'gradient' | 'atmosphere' | 0 | 1")? as u32,
            };
        }
        for (k, f) in [("clouds", &mut s.clouds), ("stars", &mut s.stars), ("skybox_blend", &mut s.skybox_blend), ("skybox_night_blend", &mut s.skybox_night_blend), ("moon", &mut s.moon_intensity)] {
            if let Some(v) = a.get(k).and_then(|v| v.as_f64()) {
                *f = v as f32;
            }
        }
        if let Some(url) = a.get("skybox").and_then(|v| v.as_str()) {
            self.load_skybox(url);
        }
        if a.get("skybox").is_some_and(|v| v.is_null()) {
            self.renderer.set_skybox(None);
        }
        // apply now so the answer reflects the change (paused games render but do not step)
        let t = self.time.sim_time;
        let r = &mut self.renderer;
        update(&mut r.settings, &mut r.sky, t);
        let s = &r.settings;
        let el = |v: Vec3| (v.y.clamp(-1.0, 1.0).asin().to_degrees() * 10.0).round() / 10.0;
        Ok(json!({
            "mode": if s.sky_mode == 1 { "atmosphere" } else { "gradient" },
            "time_of_day": s.time_of_day, "day_cycle": s.day_cycle, "day_length_minutes": s.day_length_minutes,
            "sun_elevation_deg": el(r.sky.sun), "moon_elevation_deg": el(r.sky.moon),
            "daylight": r.sky.daylight, "night": r.sky.night, "light": r.sky.light,
            "light_intensity": s.sun_intensity, "light_color": s.sun_color, "ambient": s.sky_color,
            "exposure_mul": r.sky.exposure_mul, "clouds": s.clouds, "stars": s.stars,
            "skybox": r.skybox().map(|t| t.0), "skybox_blend": s.skybox_blend, "skybox_night_blend": s.skybox_night_blend,
        }))
    }
}
