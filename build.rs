// build.rs

fn main() {
    let _current_dir = std::env::current_dir().expect("could not get current directory");

    let eigen_include_path_str = "-I/usr/local/include/eigen3";

    let mut build = cxx_build::bridge("src/glue.rs");
    if build.is_flag_supported("-fconcepts").unwrap() {
        build
            .file("src/sophus_wrapper.cc")
            .flag(eigen_include_path_str)
            .flag("-std=gnu++17")
            .flag("-fconcepts")
            .compile("sophus-rs");
    } else {
        build
            .file("src/sophus_wrapper.cc")
            .flag(eigen_include_path_str)
            .flag("-std=c++20")
            .compile("sophus-rs");
    }

    println!("cargo:rustc-link-lib=fmt");
    println!("cargo:rustc-link-lib=farm_ng_core_logging");
    println!("cargo:rustc-link-lib=sophus_image");

    println!("cargo:rerun-if-changed=src/glue.rs");
    println!("cargo:rerun-if-changed=src/sophus_wrapper.cc");
    println!("cargo:rerun-if-changed=include/sophus_wrapper.h");
    println!("cargo:rustc-env=LD_LIBRARY_PATH=/usr/local/lib");
}
