// C++ (libtorch) inference path for GPU-CorruptNet.
//
// Loads a TorchScript model, runs the corruption classifier, and reports latency —
// the throughput-critical serving path a GPU company cares about, benchmarked against
// the Python path. On AMD/ROCm, torch::kCUDA transparently maps to HIP (ROCm's C++
// runtime/kernel language), so the same code targets AMD GPUs.
//
// Build: see cpp/README.md  |  Run: ./infer model_ts.pt [iters]

#include <torch/script.h>

#include <chrono>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: infer <model_ts.pt> [iters]\n";
    return 1;
  }
  const int iters = argc > 2 ? std::stoi(argv[2]) : 100;

  torch::jit::script::Module model;
  try {
    model = torch::jit::load(argv[1]);
  } catch (const c10::Error& e) {
    std::cerr << "failed to load model: " << e.what() << "\n";
    return 1;
  }
  model.eval();
  torch::NoGradGuard no_grad;

  // CPU here; on ROCm build, torch::kCUDA maps to HIP for AMD GPUs.
  const auto device = torch::kCPU;
  model.to(device);

  auto input = torch::randn({1, 3, 224, 224}, device);
  std::vector<torch::jit::IValue> inputs{input};

  for (int i = 0; i < 20; ++i) model.forward(inputs);  // warmup

  const auto t0 = std::chrono::high_resolution_clock::now();
  torch::Tensor out;
  for (int i = 0; i < iters; ++i) out = model.forward(inputs).toTensor();
  const auto t1 = std::chrono::high_resolution_clock::now();

  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
  const auto probs = torch::sigmoid(out);
  std::cout << "C++ libtorch inference: " << ms << " ms/frame over " << iters << " iters\n";
  std::cout << "output shape: [" << out.size(0) << ", " << out.size(1) << "]\n";
  std::cout << "max corruption prob: " << probs.max().item<float>() << "\n";
  return 0;
}
