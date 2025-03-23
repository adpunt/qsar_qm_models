use std::env;

fn main() {
    let conda_prefix = env::var("CONDA_PREFIX")
        .expect("CONDA_PREFIX not set. Make sure you're in the right conda environment.");

    let include_path = format!("{}/include", conda_prefix);
    let rdkit_include = format!("{}/include/rdkit", conda_prefix);
    let lib_path = format!("{}/lib", conda_prefix);

    println!("cargo:rustc-link-search=native={}", lib_path);

    // ️ Use dynamic linking — remove "static="
    println!("cargo:rustc-link-lib=RDKitRDGeneral");
    println!("cargo:rustc-link-lib=RDKitGraphMol");
    println!("cargo:rustc-link-lib=RDKitSmilesParse");
    println!("cargo:rustc-link-lib=RDKitFileParsers");
    println!("cargo:rustc-link-lib=stdc++");

    println!("cargo:include={}", include_path);
    println!("cargo:include={}", rdkit_include);
}

