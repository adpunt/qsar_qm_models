use std::env;
use std::path::PathBuf;

#[cfg(target_os = "macos")]
fn link_macos_flags() {
    println!("cargo:rustc-link-lib=c++");          // libc++
    println!("cargo:rustc-link-lib=iconv");
    println!("cargo:rustc-link-lib=System");
    println!("cargo:rustc-link-arg=-mmacosx-version-min=13.0");
    println!("cargo:rustc-link-arg=-stdlib=libc++");
}

#[cfg(target_os = "linux")]
fn link_linux_flags() {
    println!("cargo:rustc-link-lib=stdc++");       // GNU libstdc++
    // iconv may be part of glibc, but add if needed
    println!("cargo:rustc-link-lib=iconv");
}

fn main() {
    let micromamba_lib = PathBuf::from(env::var("CONDA_PREFIX").expect("CONDA_PREFIX not set"))
        .join("lib");

    println!("cargo:rustc-link-search=native={}", micromamba_lib.display());

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

    #[cfg(target_os = "macos")]
    link_macos_flags();

    #[cfg(target_os = "linux")]
    link_linux_flags();

    println!("cargo:rustc-link-lib=c");  // always safe
    println!("cargo:rustc-link-lib=m");

    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", micromamba_lib.display());
}
