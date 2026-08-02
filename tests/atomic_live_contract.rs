use noperson::live::{
    AtomicLiveEngine, FaceAssignmentInputs, FaceAssignmentPaths, FaceIdentityInput, LiveEngine,
    LiveShadowBuilder, embedding_blake3,
};

fn assert_send<T: Send>() {}

#[test]
fn live_generation_types_can_move_to_the_dedicated_worker() {
    assert_send::<LiveEngine>();
    assert_send::<LiveShadowBuilder>();
    assert_send::<AtomicLiveEngine>();
    assert_send::<FaceAssignmentPaths>();
    assert_send::<FaceAssignmentInputs>();
    assert_send::<FaceIdentityInput>();
}

#[test]
fn precomputed_embeddings_are_content_addressed_without_a_fake_image() {
    let first = vec![0.25; 512];
    let mut second = first.clone();
    second[511] = 0.5;

    assert_eq!(
        embedding_blake3(&first).unwrap(),
        embedding_blake3(&first).unwrap()
    );
    assert_ne!(
        embedding_blake3(&first).unwrap(),
        embedding_blake3(&second).unwrap()
    );
    assert!(embedding_blake3(&[0.0; 511]).is_err());
    assert!(embedding_blake3(&[f32::NAN; 512]).is_err());
}
