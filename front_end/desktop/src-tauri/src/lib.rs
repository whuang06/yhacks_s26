mod backend;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app = tauri::Builder::default()
        .setup(|app| {
            #[cfg(not(debug_assertions))]
            {
                if let Err(e) = backend::start(app.handle()) {
                    eprintln!("[CopperGolem] {e}");
                }
            }
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    app.run(|_handle, event| {
        #[cfg(not(debug_assertions))]
        if matches!(event, tauri::RunEvent::Exit) {
            backend::kill();
        }
    });
}
