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
    cargo run --example camera_sim --release --features std

mini_optics_sim:
    cargo run --example mini_optics_sim --release --features std

viewer_ex:
    cargo run --example viewer_ex --release --features std
