clippy:
    cargo clippy --tests --features std

build:
    cargo build --release --all-targets

build-std:
    cargo build --release --all-targets --features std

build-simd:
    cargo +nightly build --release --all-targets --features simd

check-wasm:
    cargo check --lib --target wasm32-unknown-unknown

test-simd:
    cargo +nightly test --release --features simd

test:
    cargo test --release --features std

format:
    pre-commit run -a
    cargo +nightly fmt

doc:
    cargo +nightly doc --no-deps --all-features
    cargo +nightly test --release --doc --all-features

camera-sim:
    cargo run --bin camera_sim --release --features std

demo:
    cargo run --bin demo --release --features std

wasm-demo:
    trunk serve crates/sophus/index.html --release

solver-bench:
    cargo run --bin solver_bench --release

ba-bench:
    cargo run --bin ba_bench --release -p sophus_opt

kb-projection-bench:
    cargo +nightly run --bin kb_projection_bench --release -p sophus_opt --features simd
