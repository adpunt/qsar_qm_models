use std::env;
use std::path::PathBuf;

fn main() {
    // Path to your micromamba env's lib directory
    let micromamba_lib = PathBuf::from(env::var("CONDA_PREFIX").expect("CONDA_PREFIX not set"))
        .join("lib");

    println!("cargo:rustc-link-search=native={}", micromamba_lib.display());

    // RDKit libraries needed
    let rdkit_libs = [
        "RDKitRDGeneral",
        "RDKitGraphMol",
        "RDKitSmilesParse",
        "RDKitFileParsers",
        "RDKitDataStructs",
        "RDKitDescriptors",
        "RDKitFingerprints",
        "RDKitMolStandardize",
        "RDKitScaffoldNetwork",
        "RDKitSubstructMatch",
    ];

    for lib in &rdkit_libs {
        println!("cargo:rustc-link-lib={}", lib);
    }

    // Link C++ standard library (Mac uses libc++, NOT stdc++)
    println!("cargo:rustc-link-lib=c++");

    // Boost dependency from RDKit
    println!("cargo:rustc-link-lib=boost_serialization");

    // System libraries
    println!("cargo:rustc-link-lib=iconv");
    println!("cargo:rustc-link-lib=System");
    println!("cargo:rustc-link-lib=c");
    println!("cargo:rustc-link-lib=m");

    // macOS-specific minimum version
    println!("cargo:rustc-link-arg=-mmacosx-version-min=13.0");

    // Force use of libc++ (important)
    println!("cargo:rustc-link-arg=-stdlib=libc++");
}

