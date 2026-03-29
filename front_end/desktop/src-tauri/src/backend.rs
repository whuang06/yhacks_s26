//! Start the FastAPI server from the bundled app (release builds only).

use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};

use tauri::{AppHandle, Manager};

static BACKEND: Mutex<Option<Child>> = Mutex::new(None);

pub fn start(app: &AppHandle) -> Result<(), String> {
    let dir = app.path().resource_dir().map_err(|e| e.to_string())?;
    if !dir.join("server.py").is_file() {
        return Err(format!(
            "server.py missing from app Resources ({}). Rebuild the app.",
            dir.display()
        ));
    }
    let python = resolve_python_executable();
    let log_path = std::env::temp_dir().join("coppergolem-backend.log");
    let log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(|e| e.to_string())?;

    let mut cmd = Command::new(&python);
    cmd.current_dir(&dir)
        .env("PYTHONPATH", dir.to_string_lossy().as_ref())
        .args([
            "-m",
            "uvicorn",
            "server:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8765",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::from(log));

    let child = cmd.spawn().map_err(|e| {
        format!(
            "Could not start Python ({python}): {e}. Install Python 3, then: \
             pip3 install -r requirements.txt"
        )
    })?;

    *BACKEND.lock().unwrap() = Some(child);

    let deadline = Duration::from_secs(45);
    let start = Instant::now();
    while start.elapsed() < deadline {
        if std::net::TcpStream::connect("127.0.0.1:8765").is_ok() {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(120));
    }

    kill();
    Err(format!(
        "Backend did not listen on port 8765. Check GEMINI_API_KEY / pip deps. Log: {}",
        log_path.display()
    ))
}

fn resolve_python_executable() -> String {
    if let Ok(p) = std::env::var("COPPERGOLEM_PYTHON") {
        if Path::new(&p.trim()).is_file() {
            return p.trim().to_string();
        }
    }
    for c in [
        "/opt/homebrew/bin/python3",
        "/usr/local/bin/python3",
        "/usr/bin/python3",
    ] {
        if Path::new(c).is_file() {
            return c.to_string();
        }
    }
    "python3".to_string()
}

pub fn kill() {
    if let Some(mut c) = BACKEND.lock().unwrap().take() {
        let _ = c.kill();
        let _ = c.wait();
    }
}
