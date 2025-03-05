fn main() {
    println!("cargo:rustc-link-search=native=/Users/apunt/miniconda3/envs/rust_py_env/lib");
    println!("cargo:rustc-link-lib=RDKitMolStandardize");
    // println!("cargo:rustc-rpath"); // Enables rpath embedding
}
