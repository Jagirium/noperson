use noperson::gpu_preview::{PreviewGeometry, PreviewRingState};

#[test]
fn preview_rows_are_rgba_and_wgpu_copy_aligned() {
    let geometry = PreviewGeometry::new(1921, 1080).unwrap();

    assert_eq!(geometry.width(), 1921);
    assert_eq!(geometry.height(), 1080);
    assert_eq!(geometry.row_bytes() % 256, 0);
    assert!(geometry.row_bytes() >= 1921 * 4);
    assert_eq!(
        geometry.buffer_size(),
        u64::from(geometry.row_bytes()) * 1080
    );
}

#[test]
fn producer_never_reuses_a_slot_until_the_consumer_releases_it() {
    let ring = PreviewRingState::new(3);
    let a = ring.acquire().unwrap();
    let b = ring.acquire().unwrap();
    let c = ring.acquire().unwrap();

    assert!(ring.acquire().is_none());
    ring.publish(a);
    assert_eq!(ring.take_latest(), Some(a));
    assert!(ring.acquire().is_none());

    ring.release(a);
    assert_eq!(ring.acquire(), Some(a));
    ring.discard_write(a);
    ring.discard_write(b);
    ring.discard_write(c);
}

#[test]
fn consumer_drops_stale_ready_slots_and_keeps_only_the_latest() {
    let ring = PreviewRingState::new(3);
    let first = ring.acquire().unwrap();
    let second = ring.acquire().unwrap();
    ring.publish(first);
    ring.publish(second);

    assert_eq!(ring.take_latest(), Some(second));
    assert!(ring.is_free(first));
    assert!(!ring.is_free(second));
    ring.release(second);
}
