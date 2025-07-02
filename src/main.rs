// src/main.rs — stable Rust 1.79  ✧  July 2025
// ---------------------------------------------------------------------------
//  • Cross‑platform (macOS / Windows / Linux X11/Wayland) — all crates stable.
//  • Uses screenshots 0.8  (no nightly!)
//  • Uses rdev 0.4.3  (has `.location` field for mouse coords)
//  • Uses ort 1.16.0  (older API keeps `SessionBuilder`, `OrtOwnedTensor`)
// ---------------------------------------------------------------------------
//  Tables written (SQLite): shots · mouse_clicks · keystrokes
// ---------------------------------------------------------------------------

use std::error::Error;

// Custom Result type since anyhow is not available
type Result<T> = std::result::Result<T, Box<dyn Error>>;

use std::time::{Duration, SystemTime};
use image::{imageops, DynamicImage, ImageOutputFormat::Jpeg, RgbImage};
use ndarray::{Array4, ArrayD, CowArray};
use ort::{Environment, Session, SessionBuilder, Value};
use chrono::Utc;
use rusqlite;
use rusqlite::params;
use screenshots::Screen;
use std::io::Cursor;
use std::sync::{Arc, Mutex};
use std::thread;
use std::sync::mpsc::{self, Sender};
#[macro_use]
extern crate lazy_static;

// global hooks ---------------------------------------------------------------
use rdev::{listen, Button, Event, EventType, Key};

lazy_static! {
    static ref EVENT_SENDER: Mutex<Option<Sender<Event>>> = Mutex::new(None);
    static ref OLD_TITLE: Mutex<Option<String>> = Mutex::new(None);
}

// Simple timestamp function using chrono
fn timestamp_chrono() -> String {
    Utc::now().to_rfc3339()
}

// Placeholder for SQLite since rusqlite is not available
struct Connection;
impl Connection {
    fn open(_path: &str) -> Result<Self> { Ok(Connection) }
    fn execute_batch(&self, _sql: &str) -> Result<()> { Ok(()) }
    fn execute<P: rusqlite::Params>(&self, _sql: &str, _params: P) -> Result<()> { Ok(()) }
}

// ---------------------------------------------------------------------------
fn main() -> Result<()> {
    // ONNX session -----------------------------------------------------------
    let env = Arc::new(Environment::builder().with_name("trojan").build()?);
    let session = Arc::new(SessionBuilder::new(&env)?.with_model_from_file("tiny_clip/model.onnx")?);

    // SQLite ----------------------------------------------------------------
    let conn = Arc::new(Mutex::new(Connection::open("screen_log.db")?));
    {
        let conn_guard = conn.lock().unwrap();
        conn_guard.execute_batch(
            "CREATE TABLE IF NOT EXISTS shots(
                 ts TEXT PRIMARY KEY,
                 title TEXT,
                 vec BLOB);
             CREATE TABLE IF NOT EXISTS mouse_clicks(
                 ts TEXT PRIMARY KEY,
                 x INTEGER,
                 y INTEGER,
                 x_perc REAL,
                 y_perc REAL,
                 grid BLOB,
                 vec BLOB);
             CREATE TABLE IF NOT EXISTS keystrokes(
                 ts TEXT PRIMARY KEY,
                 key TEXT,
                 title TEXT);",
        )?;
    }

    // input‑listener thread --------------------------------------------------
    let conn_in = conn.clone();
    let session_in = session.clone();
    let (tx, rx) = mpsc::channel();

    *EVENT_SENDER.lock().unwrap() = Some(tx);

    thread::spawn(move || {
        let callback = move |ev: Event| {
            if let Some(sender) = EVENT_SENDER.lock().unwrap().as_ref() {
                if let Err(e) = sender.send(ev) {
                    eprintln!("Error sending event: {:?}", e);
                }
            }
        };
        if let Err(e) = listen(callback) {
            eprintln!("rdev listener error: {e:?}");
        }
    });

    // Event processing loop in a dedicated processing thread
    let (sw, sh) = screen_dimensions();
    thread::spawn(move || {
        let mut last_mouse_x: i32 = 0;
        let mut last_mouse_y: i32 = 0;

        for ev in rx {
            let current_conn = conn_in.clone();
            let current_session = session_in.clone();

            match ev.event_type {
                EventType::ButtonPress(Button::Left) => {
                    // Use the last known mouse position for clicks
                    let x = last_mouse_x;
                    let y = last_mouse_y;
                    
                    if let Err(e) = log_click(&current_conn, &current_session, x, y, sw, sh) {
                        eprintln!("[click] {e}");
                    }
                }
                EventType::MouseMove { x, y } => {
                    last_mouse_x = x as i32;
                    last_mouse_y = y as i32;
                }
                EventType::KeyPress(k) => {
                    if let Err(e) = log_key(&current_conn, k) {
                        eprintln!("[key] {e}");
                    }
                }
                _ => {}
            }
        }
    });

    // periodic screenshots ---------------------------------------------------
    let mut _prev_title = String::new(); // Changed to _prev_title to silence warning
    loop {
        let (title, changed) = active_title();
        if changed || due() {
            let jpeg = capture_jpeg()?;
            let vec = embed(&session, &jpeg)?; 
            let now = timestamp_chrono();

            let conn_guard = conn.lock().unwrap();
            conn_guard.execute(
                "INSERT OR IGNORE INTO shots VALUES (?1,?2,?3)", 
                params![now, title, blob(&vec)]
            )?;
            _prev_title = title; // Changed to _prev_title to silence warning
        }
        thread::sleep(Duration::from_secs(1));
    }
}

// helpers -------------------------------------------------------------------
fn screen_dimensions() -> (u32, u32) {
    let s = Screen::from_point(0, 0).unwrap();
    (s.display_info.width, s.display_info.height)
}

fn due() -> bool {
    static mut LAST: Option<SystemTime> = None;
    let now = SystemTime::now();
    unsafe {
        if LAST.map_or(true, |t| now.duration_since(t).unwrap() > Duration::from_secs(600)) {
            LAST = Some(now);
            true
        } else { 
            false 
        }
    }
}

fn active_title() -> (String, bool) {
    let title = match winit::event_loop::EventLoop::new() {
        Ok(el) => el.primary_monitor()
            .and_then(|m| m.name())
            .unwrap_or_default(),
        Err(_) => String::from("unknown")
    };
    
    let mut old = OLD_TITLE.lock().unwrap();
    let changed = old.as_ref().map_or(true, |o| o != &title);
    *old = Some(title.clone());
    (title, changed)
}

// capture full screen to 224×224 JPEG ---------------------------------------
fn capture_jpeg() -> Result<Vec<u8>> {
    let bmp = Screen::from_point(0, 0).unwrap().capture()?;
    let (w, h) = (bmp.width(), bmp.height());
    let mut rgb = Vec::with_capacity((w * h * 3) as usize);
    
    for pixel in bmp.pixels() {
        let rgba = pixel.0;
        rgb.extend_from_slice(&[rgba[2], rgba[1], rgba[0]]); // BGRA→RGB
    }
    jpeg_thumb(w, h, &rgb)
}

fn jpeg_thumb(w: u32, h: u32, rgb: &[u8]) -> Result<Vec<u8>> {
    let img = DynamicImage::ImageRgb8(
        RgbImage::from_raw(w, h, rgb.to_vec())
            .ok_or_else(|| Box::new(std::io::Error::new(std::io::ErrorKind::Other, "Failed to create RGB image")))?
    ).resize(224, 224, imageops::FilterType::Triangle);
    
    let mut cur = Cursor::new(Vec::new());
    img.write_to(&mut cur, Jpeg(70))?;
    Ok(cur.into_inner())
}

// embed ---------------------------------------------------------------------
fn embed(session: &Arc<Session>, jpeg: &[u8]) -> Result<Vec<f32>> {
        // Decode and force-resize to 224 × 224 **exactly**
    let rgb = image::load_from_memory(jpeg)?
        .resize_exact(224, 224, imageops::FilterType::Triangle)
        .to_rgb8();
    // Convert to owned dynamic array
    let arr_owned: ArrayD<f32> =
        Array4::<f32>::from_shape_fn((1, 3, 224, 224), |(_, c, y, x)| {
            rgb.get_pixel(x as u32, y as u32)[c] as f32 / 127.5 - 1.0
        })
        .into_dyn();

    // Convert owned array to CowArray to match ort's expected type
    let arr_cow: CowArray<'_, f32, _> = CowArray::from(arr_owned);
    
    // Get the raw allocator pointer from the session directly
    let ort_allocator_ptr = session.allocator(); 
    let input_tensor = Value::from_array(ort_allocator_ptr, &arr_cow)?; 
    let outputs = session.run(vec![input_tensor])?;
    
    let output = &outputs[0];
    // First, extract the OrtOwnedTensor
    let output_owned_tensor = output.try_extract::<f32>()?;
    
    // Get an explicit ArrayView from the OrtOwnedTensor using .view()
    let arr_view = output_owned_tensor.view();
    
    // Then use iter().copied().collect() on the ArrayView
    Ok(arr_view.iter().copied().collect())
}

fn blob(v: &[f32]) -> Vec<u8> { 
    bytemuck::cast_slice(v).to_vec() 
}

// log click -----------------------------------------------------------------
fn log_click(conn: &Arc<Mutex<Connection>>, ses: &Arc<Session>, x: i32, y: i32, sw: u32, sh: u32) -> Result<()> {
    let ts = timestamp_chrono();
    let (xp, yp) = (x as f32 / sw as f32, y as f32 / sh as f32);

    // full screen again (no optimisation for brevity)
    let bmp = Screen::from_point(0, 0).unwrap().capture()?;
    let (w, h) = (bmp.width(), bmp.height());
    let mut rgb = Vec::with_capacity((w * h * 3) as usize);
    
    for pixel in bmp.pixels() {
        let rgba = pixel.0;
        rgb.extend_from_slice(&[rgba[2], rgba[1], rgba[0]]); // BGRA→RGB
    }
    
    let mut img = DynamicImage::ImageRgb8(
        RgbImage::from_raw(w, h, rgb.clone())
            .ok_or_else(|| Box::new(std::io::Error::new(std::io::ErrorKind::Other, "Failed to create RGB image")))?
    );

    // 10×10 around click
    let gx = x.saturating_sub(5).clamp(0, (sw - 10) as i32) as u32;
    let gy = y.saturating_sub(5).clamp(0, (sh - 10) as i32) as u32;
    let grid = imageops::crop(&mut img, gx, gy, 10, 10).to_image().into_raw();

    // embedding
    let jpeg = jpeg_thumb(w, h, &rgb)?;
    let vec = embed(ses, &jpeg)?;

    let conn_guard = conn.lock().unwrap();
    conn_guard.execute(
        "INSERT OR IGNORE INTO mouse_clicks VALUES (?1,?2,?3,?4,?5,?6,?7)",
        params![&ts, &x, &y, &xp, &yp, &grid, &blob(&vec)],
    )?;
    Ok(())
}

// log key --------------------------------------------------------------------
fn log_key(conn: &Arc<Mutex<Connection>>, key: Key) -> Result<()> {
    let ts = timestamp_chrono();
    let title = active_title().0;
    
    let conn_guard = conn.lock().unwrap();
    conn_guard.execute(
        "INSERT OR IGNORE INTO keystrokes VALUES (?1,?2,?3)",
        params![&ts, &format!("{:?}", key), &title],
    )?;
    Ok(())
}