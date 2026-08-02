use std::any::TypeId;
use std::path::PathBuf;

use noperson::extra_gui::{ExtraGuiApp, ExtraGuiPanel};

fn assert_eframe_app<T: eframe::App>() {}

#[test]
fn realtime_and_extra_gui_are_independent_frontend_types() {
    assert_eframe_app::<noperson::app::App>();
    assert_eframe_app::<ExtraGuiApp>();
    assert_ne!(
        TypeId::of::<noperson::app::App>(),
        TypeId::of::<ExtraGuiApp>()
    );
}

#[test]
fn extra_gui_owns_its_editor_navigation_state() {
    let mut app = ExtraGuiApp::new(PathBuf::from("models"));

    assert_eq!(app.active_panel(), ExtraGuiPanel::Media);
    app.select_panel(ExtraGuiPanel::Timeline);
    assert_eq!(app.active_panel(), ExtraGuiPanel::Timeline);
    assert_eq!(
        ExtraGuiPanel::ALL,
        [
            ExtraGuiPanel::Media,
            ExtraGuiPanel::Faces,
            ExtraGuiPanel::Preview,
            ExtraGuiPanel::Timeline,
            ExtraGuiPanel::Parameters,
            ExtraGuiPanel::Settings,
        ]
    );
}
