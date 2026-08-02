use std::collections::BTreeMap;

use noperson::extra_gui::{
    ARC_FACE_MODEL, ControlValue, EmbeddingMergeMethod, EmbeddingStore, FaceWorkspace,
    SavedEmbedding, control_catalog, load_embeddings, save_embeddings,
};

fn embedding(values: &[f32]) -> EmbeddingStore {
    BTreeMap::from([(ARC_FACE_MODEL.to_owned(), values.to_vec())])
}

#[test]
fn target_assignments_merge_sources_like_crossswap() {
    let catalog = control_catalog().unwrap();
    let mut faces = FaceWorkspace::default();
    faces
        .add_source("source-a", "A", None, embedding(&[1.0, 8.0]))
        .unwrap();
    faces
        .add_source("source-b", "B", None, embedding(&[3.0, 2.0]))
        .unwrap();
    faces
        .add_target("target", None, embedding(&[0.0, 0.0]), &catalog)
        .unwrap();
    faces.assign_source("target", "source-a", true).unwrap();
    faces.assign_source("target", "source-b", true).unwrap();

    assert_eq!(
        faces
            .assigned_embedding("target", ARC_FACE_MODEL, EmbeddingMergeMethod::Mean)
            .unwrap(),
        vec![2.0, 5.0]
    );
    // CrossSwap uses torch.median here, which selects the lower middle value.
    assert_eq!(
        faces
            .assigned_embedding("target", ARC_FACE_MODEL, EmbeddingMergeMethod::Median)
            .unwrap(),
        vec![1.0, 2.0]
    );
}

#[test]
fn saved_merged_identity_uses_numpy_even_median_semantics() {
    let mut faces = FaceWorkspace::default();
    faces
        .add_source("source-a", "A", None, embedding(&[1.0, 8.0]))
        .unwrap();
    faces
        .add_source("source-b", "B", None, embedding(&[3.0, 2.0]))
        .unwrap();

    faces
        .create_merged(
            "merged",
            "A + B",
            &["source-a".to_owned(), "source-b".to_owned()],
            EmbeddingMergeMethod::Median,
        )
        .unwrap();
    assert_eq!(
        faces.merged("merged").unwrap().embeddings[ARC_FACE_MODEL],
        vec![2.0, 5.0]
    );
}

#[test]
fn every_target_owns_parameters_and_swap_all_is_explicit() {
    let catalog = control_catalog().unwrap();
    let mut faces = FaceWorkspace::default();
    faces
        .add_source("source", "Source", None, embedding(&[4.0]))
        .unwrap();
    faces
        .add_target("first", None, embedding(&[0.0]), &catalog)
        .unwrap();
    faces
        .add_target("second", None, embedding(&[1.0]), &catalog)
        .unwrap();

    faces
        .target_mut("first")
        .unwrap()
        .controls
        .set(
            "SimilarityThresholdSlider",
            ControlValue::Slider(77.0),
            &catalog,
        )
        .unwrap();
    assert_eq!(
        faces
            .target("second")
            .unwrap()
            .controls
            .get("SimilarityThresholdSlider"),
        Some(&ControlValue::Slider(60.0))
    );

    faces.set_swap_all(true);
    faces.set_source_assignment(None, "source", true).unwrap();
    assert!(faces.swap_all());
    assert_eq!(faces.selected_target(), None);
    assert!(faces.source_is_assigned(None, "source"));
    assert_eq!(
        faces
            .swap_all_embedding(ARC_FACE_MODEL, EmbeddingMergeMethod::Mean)
            .unwrap(),
        vec![4.0]
    );
}

#[test]
fn deleting_an_identity_cleans_all_assignments() {
    let catalog = control_catalog().unwrap();
    let mut faces = FaceWorkspace::default();
    faces
        .add_source("source", "Source", None, embedding(&[1.0]))
        .unwrap();
    faces
        .add_target("target", None, embedding(&[0.0]), &catalog)
        .unwrap();
    faces.assign_source("target", "source", true).unwrap();
    assert!(faces.remove_source("source"));
    assert!(faces.target("target").unwrap().assigned_sources.is_empty());
}

#[test]
fn crosswap_embedding_files_roundtrip_without_torch() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("embeddings.json");
    let embeddings = vec![SavedEmbedding {
        name: "Hero blend".to_owned(),
        embedding_store: embedding(&[0.25, 0.75]),
    }];

    save_embeddings(&path, &embeddings).unwrap();
    assert_eq!(load_embeddings(&path).unwrap(), embeddings);
}

#[test]
fn clearing_merged_embeddings_cleans_target_and_swap_all_routes() {
    let catalog = control_catalog().unwrap();
    let mut faces = FaceWorkspace::default();
    faces
        .add_merged_store("merged", "Merged", embedding(&[1.0]))
        .unwrap();
    faces
        .add_target("target", None, embedding(&[0.0]), &catalog)
        .unwrap();
    faces.assign_merged("target", "merged", true).unwrap();
    faces.set_merged_assignment(None, "merged", true).unwrap();

    assert_eq!(faces.clear_merged(), 1);
    assert!(faces.target("target").unwrap().assigned_merged.is_empty());
    assert!(!faces.merged_is_assigned(None, "merged"));
}

#[test]
fn clearing_target_scan_state_preserves_source_identity_library() {
    let catalog = control_catalog().unwrap();
    let mut faces = FaceWorkspace::default();
    faces
        .add_source("source", "Source", None, embedding(&[1.0]))
        .unwrap();
    faces
        .add_target("target", None, embedding(&[0.0]), &catalog)
        .unwrap();

    assert_eq!(faces.clear_targets(), 1);
    assert_eq!(faces.targets().count(), 0);
    assert!(faces.source("source").is_some());
}
