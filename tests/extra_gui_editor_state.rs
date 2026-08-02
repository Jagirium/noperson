use std::path::{Path, PathBuf};

use noperson::extra_gui::{
    EditorTimeline, MediaKind, MediaLibrary, MediaRole, PreviewViewport, control_catalog,
    discover_media,
};

#[test]
fn media_library_remembers_the_first_picker_directory_for_the_second() {
    let mut library = MediaLibrary::default();
    let target = library
        .add(MediaRole::Target, PathBuf::from("/work/shoot/target.png"))
        .expect("target image");

    assert_eq!(library.last_directory(), Some(Path::new("/work/shoot")));
    assert_eq!(library.selected(MediaRole::Target), Some(target));

    let source = library
        .add(MediaRole::Source, PathBuf::from("/work/faces/source.webp"))
        .expect("source image");
    assert_eq!(library.last_directory(), Some(Path::new("/work/faces")));
    assert_eq!(library.selected(MediaRole::Source), Some(source));
    assert_eq!(library.item(target).unwrap().kind, MediaKind::Image);
}

#[test]
fn media_library_accepts_video_targets_but_not_video_source_faces() {
    let mut library = MediaLibrary::default();
    let video = library
        .add(MediaRole::Target, PathBuf::from("clip.mkv"))
        .expect("video target");
    assert_eq!(library.item(video).unwrap().kind, MediaKind::Video);
    assert!(
        library
            .add(MediaRole::Source, PathBuf::from("face.mp4"))
            .is_err()
    );
}

#[test]
fn preview_viewport_has_browser_style_bounded_zoom() {
    let mut viewport = PreviewViewport::default();
    assert_eq!(viewport.zoom(), 1.0);
    viewport.zoom_by(2.0, [640.0, 360.0]);
    assert_eq!(viewport.zoom(), 2.0);
    assert_eq!(viewport.anchor(), [640.0, 360.0]);

    for _ in 0..20 {
        viewport.zoom_by(2.0, [0.0, 0.0]);
    }
    assert_eq!(viewport.zoom(), PreviewViewport::MAX_ZOOM);
    viewport.reset();
    assert_eq!(viewport.zoom(), 1.0);
    assert_eq!(viewport.pan(), [0.0, 0.0]);
}

#[test]
fn timeline_markers_capture_controls_and_support_nearest_navigation() {
    let catalog = control_catalog().expect("catalog");
    let controls = noperson::extra_gui::ControlState::from_catalog(&catalog).unwrap();
    let mut timeline = EditorTimeline::new(300, 30.0);
    timeline.seek(40);
    timeline.add_marker(controls.clone());
    timeline.seek(180);
    timeline.add_marker(controls);

    timeline.seek(100);
    assert_eq!(timeline.previous_marker(), Some(40));
    assert_eq!(timeline.next_marker(), Some(180));
    assert!(timeline.marker_at(40).is_some());
    timeline.remove_marker(40);
    assert!(timeline.marker_at(40).is_none());
}

#[test]
fn media_folder_discovery_is_sorted_filtered_and_optionally_recursive() {
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(directory.path().join("b.mp4"), []).unwrap();
    std::fs::write(directory.path().join("a.png"), []).unwrap();
    std::fs::write(directory.path().join("ignored.txt"), []).unwrap();
    std::fs::create_dir(directory.path().join("nested")).unwrap();
    std::fs::write(directory.path().join("nested/face.webp"), []).unwrap();

    let flat = discover_media(directory.path(), MediaRole::Target, false).unwrap();
    assert_eq!(flat.len(), 2);
    assert!(flat[0].ends_with("a.png"));
    assert!(flat[1].ends_with("b.mp4"));

    let recursive = discover_media(directory.path(), MediaRole::Source, true).unwrap();
    assert_eq!(recursive.len(), 2);
    assert!(
        recursive
            .iter()
            .all(|path| path.extension().unwrap() != "mp4")
    );
}

#[test]
fn clearing_one_media_role_preserves_the_other_role() {
    let mut library = MediaLibrary::default();
    let target = library
        .add(MediaRole::Target, PathBuf::from("target.png"))
        .unwrap();
    let source = library
        .add(MediaRole::Source, PathBuf::from("source.png"))
        .unwrap();

    assert_eq!(library.clear(MediaRole::Source), vec![source]);
    assert!(library.item(source).is_none());
    assert!(library.item(target).is_some());
}
