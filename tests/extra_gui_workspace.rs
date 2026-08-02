use std::path::Path;

use noperson::extra_gui::{
    ARC_FACE_MODEL, ControlValue, MediaRole, WorkspaceDocument, control_catalog,
};
use serde_json::json;

#[test]
fn crosswap_workspace_import_preserves_media_selection_and_controls() {
    let catalog = control_catalog().unwrap();
    let legacy = json!({
        "selected_media_id": "target-b",
        "target_medias_data": [
            {"media_id": "target-a", "media_path": "/shots/a.png"},
            {"media_id": "target-b", "media_path": "/shots/b.mkv"}
        ],
        "input_faces_data": {
            "source-a": {"media_path": "/faces/a.webp"}
        },
        "embeddings_data": {
            "blend-a": {
                "embedding_name": "Hero blend",
                "embedding_store": {"Inswapper128ArcFace": [0.25, 0.75]}
            }
        },
        "target_faces_data": {
            "face-a": {
                "cropped_face": [[[1, 2, 3], [4, 5, 6]]],
                "embedding_store": {"Inswapper128ArcFace": [0.1, 0.2]},
                "parameters": {"SimilarityThresholdSlider": 66},
                "control": {"DetectorScoreSlider": 71},
                "assigned_input_faces": ["source-a"],
                "assigned_merged_embeddings": ["blend-a"],
                "assigned_input_embedding": {"Inswapper128ArcFace": [0.2, 0.8]}
            }
        },
        "control": {
            "ProvidersPrioritySelection": "TensorRT",
            "DetectorScoreSlider": 73,
            "AutoSwapToggle": true,
            "OutputMediaFolder": "/renders"
        },
        "current_widget_parameters": {
            "SimilarityThresholdSlider": 81,
            "FaceRestorerEnableToggle": true,
            "FaceRestorerTypeSelection": "GPEN-512"
        },
        "markers": {}
    });

    let workspace = WorkspaceDocument::from_crossswap_value("legacy", legacy, &catalog)
        .expect("CrossSwap workspace import");
    let selected = workspace.media.selected(MediaRole::Target).unwrap();
    assert_eq!(
        workspace.media.item(selected).unwrap().path,
        Path::new("/shots/b.mkv")
    );
    assert_eq!(workspace.media.items(MediaRole::Source).count(), 1);
    let target_face = workspace.faces.target("face-a").unwrap();
    assert_eq!(
        target_face.crop.as_ref().unwrap().rgb,
        vec![1, 2, 3, 4, 5, 6]
    );
    assert!(target_face.assigned_sources.contains("source-a"));
    assert!(target_face.assigned_merged.contains("blend-a"));
    assert_eq!(
        target_face.assigned_embedding_cache[ARC_FACE_MODEL],
        vec![0.2, 0.8]
    );
    assert_eq!(
        target_face.controls.get("SimilarityThresholdSlider"),
        Some(&ControlValue::Slider(66.0))
    );
    assert!(workspace.faces.merged("blend-a").is_some());
    assert_eq!(
        workspace.controls.get("ProvidersPrioritySelection"),
        Some(&ControlValue::Choice("TensorRT".to_owned()))
    );
    assert_eq!(
        workspace.controls.get("SimilarityThresholdSlider"),
        Some(&ControlValue::Slider(81.0))
    );
    assert_eq!(
        workspace.output_directory.as_deref(),
        Some(Path::new("/renders"))
    );
}

#[test]
fn native_workspace_is_atomic_and_roundtrips_without_catalog_data() {
    let catalog = control_catalog().unwrap();
    let mut workspace = WorkspaceDocument::new("grade session", &catalog).unwrap();
    workspace
        .media
        .add(MediaRole::Target, "/shots/input.png".into())
        .unwrap();
    workspace
        .controls
        .set("ColorEnableToggle", ControlValue::Toggle(true), &catalog)
        .unwrap();

    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("session.noperson.json");
    workspace.save_atomic(&path).unwrap();
    assert!(!directory.path().join("session.noperson.json.tmp").exists());

    let restored = WorkspaceDocument::load(&path, &catalog).unwrap();
    assert_eq!(restored.name, "grade session");
    assert_eq!(restored.media, workspace.media);
    assert_eq!(restored.controls, workspace.controls);
}

#[test]
fn crosswap_export_uses_original_control_keys_and_plain_values() {
    let catalog = control_catalog().unwrap();
    let mut workspace = WorkspaceDocument::new("export", &catalog).unwrap();
    workspace
        .controls
        .set(
            "ColorGammaDecimalSlider",
            ControlValue::Slider(1.25),
            &catalog,
        )
        .unwrap();

    let value = workspace.to_crossswap_value(&catalog).unwrap();
    assert_eq!(
        value["current_widget_parameters"]["ColorGammaDecimalSlider"],
        1.25
    );
    assert_eq!(value["control"]["ProvidersPrioritySelection"], "CUDA");
    assert!(value.get("noperson_catalog").is_none());
}

#[test]
fn crossswap_face_graph_roundtrips_without_collapsing_per_face_state() {
    let catalog = control_catalog().unwrap();
    let legacy = json!({
        "selected_media_id": false,
        "target_medias_data": [],
        "input_faces_data": {"source": {"media_path": "/faces/a.png"}},
        "embeddings_data": {},
        "target_faces_data": {
            "target": {
                "cropped_face": [[[10, 20, 30]]],
                "embedding_store": {"Inswapper128ArcFace": [0.1]},
                "parameters": {"SimilarityThresholdSlider": 91},
                "control": {},
                "assigned_input_faces": ["source"],
                "assigned_merged_embeddings": [],
                "assigned_input_embedding": {"Inswapper128ArcFace": [0.9]}
            }
        },
        "markers": {},
        "control": {},
        "current_widget_parameters": {}
    });
    let workspace = WorkspaceDocument::from_crossswap_value("faces", legacy, &catalog).unwrap();
    let exported = workspace.to_crossswap_value(&catalog).unwrap();

    assert_eq!(
        exported["target_faces_data"]["target"]["parameters"]["SimilarityThresholdSlider"],
        91.0
    );
    assert_eq!(
        exported["target_faces_data"]["target"]["assigned_input_faces"][0],
        "source"
    );
    assert_eq!(
        exported["target_faces_data"]["target"]["cropped_face"],
        json!([[[10, 20, 30]]])
    );
}

#[test]
fn crossswap_markers_preserve_every_face_parameter_snapshot() {
    let catalog = control_catalog().unwrap();
    let face = |embedding: f64| {
        json!({
            "cropped_face": [[[1, 2, 3]]],
            "embedding_store": {"Inswapper128ArcFace": [embedding]},
            "parameters": {},
            "control": {},
            "assigned_input_faces": [],
            "assigned_merged_embeddings": [],
            "assigned_input_embedding": {}
        })
    };
    let legacy = json!({
        "selected_media_id": false,
        "target_medias_data": [],
        "input_faces_data": {},
        "embeddings_data": {},
        "target_faces_data": {"face-a": face(0.1), "face-b": face(0.2)},
        "control": {},
        "current_widget_parameters": {},
        "markers": {
            "12": {
                "control": {"DetectorScoreSlider": 79},
                "parameters": {
                    "face-a": {"SimilarityThresholdSlider": 33},
                    "face-b": {"SimilarityThresholdSlider": 88}
                }
            }
        }
    });

    let workspace = WorkspaceDocument::from_crossswap_value("markers", legacy, &catalog).unwrap();
    let exported = workspace.to_crossswap_value(&catalog).unwrap();
    assert_eq!(
        exported["markers"]["12"]["parameters"]["face-a"]["SimilarityThresholdSlider"],
        33.0
    );
    assert_eq!(
        exported["markers"]["12"]["parameters"]["face-b"]["SimilarityThresholdSlider"],
        88.0
    );
    assert_eq!(
        exported["markers"]["12"]["control"]["DetectorScoreSlider"],
        79.0
    );
}
