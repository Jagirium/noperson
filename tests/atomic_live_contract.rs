use noperson::live::{AtomicLiveEngine, LiveEngine, LiveShadowBuilder};

fn assert_send<T: Send>() {}

#[test]
fn live_generation_types_can_move_to_the_dedicated_worker() {
    assert_send::<LiveEngine>();
    assert_send::<LiveShadowBuilder>();
    assert_send::<AtomicLiveEngine>();
}
