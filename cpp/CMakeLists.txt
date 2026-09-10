cmake_minimum_required(VERSION 3.18)
project(gpu_corruptnet_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Uses the libtorch shipped inside the installed PyTorch wheel (exact ABI match):
#   cmake -B build -DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)') .
find_package(Torch REQUIRED)

add_executable(infer infer.cpp)
target_link_libraries(infer "${TORCH_LIBRARIES}")
