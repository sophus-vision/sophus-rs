#![cfg(feature = "std")]

fn main() {
    env_logger::init();
    pollster::block_on(sophus::examples::camera_sim::run_offscreen());
}
