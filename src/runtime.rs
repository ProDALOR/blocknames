//! Locate optional CUDA runtime libraries without launcher scripts or global PATH
//! changes. Call once at process startup, before Rayon/prefetch threads are started.
#[cfg(windows)]
use anyhow::ensure;
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

pub fn data_dir() -> Result<PathBuf> {
    if cfg!(windows) {
        Ok(PathBuf::from(
            std::env::var_os("LOCALAPPDATA")
                .context("set --output when LOCALAPPDATA is unavailable")?,
        )
        .join("blocknames"))
    } else {
        let base = std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|d| PathBuf::from(d).join(".local/share")))
            .context("set --output when a user data directory is unavailable")?;
        Ok(base.join("blocknames"))
    }
}

pub fn kernel_cache_dir() -> Option<PathBuf> {
    if cfg!(windows) {
        data_dir().ok().map(|d| d.join("kernels"))
    } else {
        std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|d| PathBuf::from(d).join(".cache")))
            .map(|d| d.join("blocknames/kernels"))
    }
}

fn library_in(root: &Path) -> Option<PathBuf> {
    let folders = [
        root.to_path_buf(),
        root.join("bin"),
        root.join("bin/x64"),
        root.join("lib64"),
        root.join("lib"),
    ];
    let names: &[&str] = if cfg!(windows) {
        &["nvrtc64_130_0.dll"]
    } else {
        &["libnvrtc.so.13", "libnvrtc.so"]
    };
    folders.into_iter().find_map(|folder| {
        names
            .iter()
            .map(|name| folder.join(name))
            .find(|p| p.is_file())
    })
}

fn cuda13_version(name: &str) -> Option<Vec<u32>> {
    let name = name
        .strip_prefix('v')
        .or_else(|| name.strip_prefix("cuda-"))
        .unwrap_or(name);
    let version: Vec<u32> = name
        .split('.')
        .map(str::parse)
        .collect::<std::result::Result<_, _>>()
        .ok()?;
    (version.first() == Some(&13)).then_some(version)
}

fn versioned_toolkits(parent: &Path) -> Vec<PathBuf> {
    let mut versions: Vec<_> = std::fs::read_dir(parent)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let version = cuda13_version(entry.file_name().to_str()?)?;
            entry.path().is_dir().then_some((version, entry.path()))
        })
        .collect();
    versions.sort_by(|a, b| b.0.cmp(&a.0));
    versions.into_iter().map(|(_, path)| path).collect()
}

fn discovery_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Ok(exe) = std::env::current_exe()
        && let Some(parent) = exe.parent()
    {
        roots.push(parent.to_path_buf());
        roots.push(parent.join("cuda"));
    }
    for name in ["CUDA_PATH", "CUDA_HOME"] {
        if let Some(path) = std::env::var_os(name) {
            roots.push(path.into());
        }
    }
    // CUDA_PATH may point at another major version when multiple toolkits are
    // installed. Also inspect the version-specific installer environment values.
    let mut versions: Vec<_> = std::env::vars_os()
        .filter_map(|(key, path)| {
            let key = key.to_str()?.to_ascii_uppercase();
            let version = cuda13_version(&key.strip_prefix("CUDA_PATH_V")?.replace('_', "."))?;
            Some((version, PathBuf::from(path)))
        })
        .collect();
    versions.sort_by(|a, b| b.0.cmp(&a.0));
    roots.extend(versions.into_iter().map(|(_, path)| path));
    if cfg!(windows) {
        for name in ["ProgramFiles", "ProgramW6432"] {
            if let Some(path) = std::env::var_os(name) {
                roots.extend(versioned_toolkits(
                    &PathBuf::from(path).join("NVIDIA GPU Computing Toolkit/CUDA"),
                ));
            }
        }
    } else {
        roots.extend([PathBuf::from("/usr/local/cuda"), PathBuf::from("/opt/cuda")]);
        roots.extend(versioned_toolkits(Path::new("/usr/local")));
    }
    if let Ok(data) = data_dir() {
        roots.push(data.join("cuda"));
    }
    for name in ["PATH", "LD_LIBRARY_PATH"] {
        if let Some(paths) = std::env::var_os(name) {
            roots.extend(std::env::split_paths(&paths));
        }
    }
    roots
}

/// Retain explicitly loaded shared libraries until all CUDA objects have dropped.
pub struct CudaRuntime {
    pub library: Option<PathBuf>,
    _libraries: Vec<libloading::Library>,
}

pub fn configure(explicit: Option<&Path>) -> Result<CudaRuntime> {
    let chosen = if let Some(path) = explicit {
        Some(library_in(path).with_context(||format!("no CUDA 13 NVRTC library found in {} (expected a DLL/lib directory or CUDA toolkit root)",path.display()))?)
    } else {
        discovery_roots()
            .into_iter()
            .find_map(|root| library_in(&root))
    };
    let library = chosen.map(|p| p.canonicalize()).transpose()?;
    let mut libraries = Vec::new();
    if let Some(path) = &library {
        #[cfg(windows)]
        {
            use std::os::windows::ffi::OsStrExt;
            #[link(name = "kernel32")]
            unsafe extern "system" {
                fn SetDllDirectoryW(path: *const u16) -> i32;
            }
            let folder: Vec<u16> = path
                .parent()
                .unwrap()
                .as_os_str()
                .encode_wide()
                .chain(Some(0))
                .collect();
            // No threads or CUDA bindings have been started yet. This changes
            // this process's DLL search directory, never the user's system PATH.
            ensure!(
                unsafe { SetDllDirectoryW(folder.as_ptr()) } != 0,
                "cannot configure NVRTC DLL directory"
            );
            libraries.push(
                unsafe { libloading::Library::new(path) }
                    .context("loading NVRTC; keep its matching builtins DLL beside it")?,
            );
        }
        #[cfg(unix)]
        {
            // dlopen by absolute path makes --cuda-path work without relying on
            // LD_LIBRARY_PATH changes after the ELF loader has initialized.
            let library = unsafe {
                libloading::os::unix::Library::open(
                    Some(path),
                    libloading::os::unix::RTLD_NOW | libloading::os::unix::RTLD_GLOBAL,
                )
            }
            .context("loading NVRTC")?;
            libraries.push(library.into());
        }
    }
    Ok(CudaRuntime {
        library,
        _libraries: libraries,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    #[test]
    fn detects_versioned_toolkits_numerically_and_nested_library_directories() {
        let root = std::env::temp_dir().join(format!(
            "blocknames-cuda-discovery-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        for name in ["v12.9", "v13.2", "v13.10", "v14.0"] {
            fs::create_dir_all(root.join(name)).unwrap();
        }
        let folder = root.join("v13.10/bin/x64");
        fs::create_dir_all(&folder).unwrap();
        let name = if cfg!(windows) {
            "nvrtc64_130_0.dll"
        } else {
            "libnvrtc.so.13"
        };
        let library = folder.join(name);
        fs::write(&library, []).unwrap();
        assert_eq!(
            versioned_toolkits(&root),
            vec![root.join("v13.10"), root.join("v13.2")]
        );
        assert_eq!(library_in(&root.join("v13.10")), Some(library.clone()));
        assert!(library_in(&root.join("v13.2")).is_none());
        fs::remove_file(library).unwrap();
        fs::remove_dir(&folder).unwrap();
        fs::remove_dir(folder.parent().unwrap()).unwrap();
        for name in ["v12.9", "v13.2", "v13.10", "v14.0"] {
            fs::remove_dir(root.join(name)).unwrap();
        }
        fs::remove_dir(root).unwrap();
    }
}
