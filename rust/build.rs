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
    // Re-run this script when the active environment changes. Without it cargo
    // keeps the -L path from whenever the script last ran, so a rebuilt or
    // moved environment links against a directory that is no longer there and
    // the failure arrives as forty lines of "cannot find -lRDKit...".
    println!("cargo:rerun-if-env-changed=CONDA_PREFIX");

    let prefix = match env::var("CONDA_PREFIX") {
        Ok(p) => p,
        Err(_) => panic!(
            "CONDA_PREFIX is not set, and it is the only place this build looks \
             for RDKit. Activate the environment first:\n\
             \n\
             \x20   source \"$(conda info --base)/etc/profile.d/conda.sh\"\n\
             \x20   conda activate env_test\n\
             \x20   cd rust && cargo build --release\n"
        ),
    };
    let micromamba_lib = PathBuf::from(&prefix).join("lib");

    // Name the real problem here rather than letting the linker report it ten
    // times over. On 2026-08-28 this failed with "cannot find -lRDKitRDGeneral"
    // because the environment the path points at had been deleted.
    let probe_so = micromamba_lib.join("libRDKitRDGeneral.so");
    let probe_dylib = micromamba_lib.join("libRDKitRDGeneral.dylib");
    if !probe_so.exists() && !probe_dylib.exists() {
        // Tell the two cases apart. A conda RDKit ships versioned files, and
        // `-lRDKitRDGeneral` needs the unversioned name: setup.sh creates those
        // symlinks on linux, so "the versioned files are there" means setup.sh
        // has not been sourced, not that RDKit is missing.
        let versioned = std::fs::read_dir(&micromamba_lib)
            .map(|d| {
                d.filter_map(|e| e.ok())
                    .any(|e| e.file_name().to_string_lossy().starts_with("libRDKitRDGeneral."))
            })
            .unwrap_or(false);
        if versioned {
            panic!(
                "RDKit is in {} but only under its versioned file names, and\n\
                 `-lRDKitRDGeneral` needs the unversioned symlink. setup.sh\n\
                 creates those. Source it and build again:\n\
                 \n\
                 \x20   . ./setup.sh\n\
                 \x20   cd rust && cargo build --release\n",
                micromamba_lib.display()
            );
        }
        panic!(
            "RDKit is not in the active environment, so this cannot link.\n\
             \x20 looked in: {}\n\
             \x20 CONDA_PREFIX: {}\n\
             \n\
             Either the wrong environment is active, or it does not have\n\
             rdkit-dev in it. env_test is the one that does. If env_test is\n\
             missing entirely, put it back first -- it takes no solve:\n\
             \n\
             \x20   REBUILD_RESTORE_ONLY=1 bash scripts/rebuild_env.sh\n\
             \n\
             then activate it and build again.\n",
            micromamba_lib.display(),
            prefix
        );
    }

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
