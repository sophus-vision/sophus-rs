fn main() {
    println!("cargo:rustc-check-cfg=cfg(nightly)"); // Declares 'nightly' as a valid cfg condition

    let is_nightly =
        rustc_version::version_meta().unwrap().channel == rustc_version::Channel::Nightly;

    if is_nightly {
        println!("cargo:rustc-cfg=nightly");
    }
}
