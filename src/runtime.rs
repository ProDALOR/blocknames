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

/// Retain explicitly loaded shared libraries until all CUDA objects have dropped.
pub struct CudaRuntime {
    _libraries: Vec<libloading::Library>,
}

pub fn configure(explicit: Option<&Path>) -> Result<CudaRuntime> {
    let chosen = if let Some(path) = explicit {
        Some(library_in(path).with_context(||format!("no CUDA 13 NVRTC library found in {} (expected a DLL/lib directory or CUDA toolkit root)",path.display()))?)
    } else {
        let mut roots = Vec::new();
        if let Ok(exe) = std::env::current_exe()
            && let Some(parent) = exe.parent()
        {
            roots.push(parent.to_path_buf());
            roots.push(parent.join("cuda"));
        }
        if let Some(path) = std::env::var_os("CUDA_PATH") {
            roots.push(path.into());
        }
        if let Ok(data) = data_dir() {
            roots.push(data.join("cuda"));
        }
        roots.push(PathBuf::from("/usr/local/cuda"));
        roots.into_iter().find_map(|root| library_in(&root))
    };
    let mut libraries = Vec::new();
    if let Some(library) = chosen {
        let path = library.canonicalize()?;
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
                unsafe { libloading::Library::new(&path) }
                    .context("loading NVRTC; keep its matching builtins DLL beside it")?,
            );
        }
        #[cfg(unix)]
        {
            // dlopen by absolute path makes --cuda-path work without relying on
            // LD_LIBRARY_PATH changes after the ELF loader has initialized.
            let library = unsafe {
                libloading::os::unix::Library::open(
                    Some(&path),
                    libloading::os::unix::RTLD_NOW | libloading::os::unix::RTLD_GLOBAL,
                )
            }
            .context("loading NVRTC")?;
            libraries.push(library.into());
        }
    }
    Ok(CudaRuntime {
        _libraries: libraries,
    })
}
