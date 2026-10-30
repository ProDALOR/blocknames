# Windows CUDA runtime bundle

Release ZIPs contain `cuda/nvrtc64_130_0.dll`, its matching
`cuda/nvrtc-builtins64_130.dll`, and the upstream `cuda/LICENSE.NVIDIA.txt`.
The application loads these libraries automatically from beside the executable.
Keep the entire extracted ZIP together. A compatible NVIDIA GPU and driver remain
required; the bundle does not install a driver or change system PATH.

The release workflow downloads this immutable NVIDIA archive:

```text
https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvrtc/windows-x86_64/cuda_nvrtc-windows-x86_64-13.0.88-archive.zip
SHA256: 8c50a52467826167e0dbe99936140c52d62272bfc5849fe2d6587d050c8c5d29
```

Version and digest are listed in NVIDIA's [CUDA 13.0.2 redistributable manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.0.2.json).
Only the two runtime DLLs and their complete license are extracted; headers, static
libraries, alternate runtimes, and developer tools are excluded. The source archive
digest is checked before extraction, and release `SHA256SUMS` covers the final ZIP.
NVIDIA components remain subject to the included NVIDIA license and may only be
used as permitted by it; this project's MIT license covers the project's own code.

The standalone `.exe` release asset and source builds require an existing NVRTC
installation or a copy of the `cuda` folder. Neither compiles a dependency on CUDA
headers or import libraries into the Rust build.

Release CI checks that the packaged application loads its bundled NVRTC even on a
CPU-only runner. GPU validation runs the actual extracted ZIP on an RTX 5090 with a
fresh data directory and no CUDA environment overrides, and checks the loaded DLL
path points into that extracted ZIP.
