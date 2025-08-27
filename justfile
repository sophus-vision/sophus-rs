clippy:
    cargo clippy

build:
    cargo build --release --all-targets

build-std:
    cargo build --release --all-targets --features std

build-simd:
    cargo +nightly build --release --all-targets --features simd

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

camera_sim:
    cargo run --bin camera_sim --release --features std

demo:
    cargo run --bin demo --release --features std

wasm_demo:
    trunk serve crates/sophus/index.html --release

bench:
    cargo run --bin solver_bench --release
