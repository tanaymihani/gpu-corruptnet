# C++ / libtorch inference path (M6)

A native C++ inference path for the corruption classifier — the throughput-critical
serving path, benchmarked against Python. Links against the **libtorch that ships inside
the installed PyTorch wheel**, so there's no separate download and the ABI matches exactly.

## Measured (Apple M2, CPU, ResNet-50 @ 224px)

| Path | latency |
|---|---|
| Python TorchScript | 30.7 ms/frame |
| **C++ libtorch** | **21.7 ms/frame** (~**1.4× faster**) |

The C++ path drops Python interpreter overhead. On GPU the gap widens further.

## Build & run

```bash
# 1. Export a trained checkpoint to TorchScript
python scripts/export_torchscript.py model_resnet50_<stamp>.pt --out cpp/model_ts.pt

# 2. Configure against the installed libtorch, then build
cd cpp
cmake -B build -DCMAKE_PREFIX_PATH=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)') .
cmake --build build

# 3. Run (point the loader at libtorch's dylibs)
DYLD_LIBRARY_PATH=$(python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))') \
  ./build/infer model_ts.pt 200
```

> On some macOS preview toolchains, clang doesn't auto-add the SDK's libc++ headers; if you
> hit `'cassert' file not found`, add
> `-DCMAKE_CXX_FLAGS="-isystem $(xcrun --show-sdk-path)/usr/include/c++/v1"` to the configure step.

## AMD / ROCm / HIP

AMD GPUs are programmed with **HIP** — a C++ runtime API and kernel language with a CUDA-like
interface. PyTorch's ROCm build transparently maps `torch::kCUDA` (and `torch.cuda.*`) onto HIP,
so **this same C++ file targets AMD hardware unchanged**: build it against a ROCm libtorch and
switch `device` to `torch::kCUDA` — the CUDA calls hipify to HIP at build time. A custom
preprocessing op could likewise be pushed into a HIP/C++ extension via PyTorch's
CUDA-extension mechanism, which hipifies on ROCm.
