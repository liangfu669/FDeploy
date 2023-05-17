project(common)

add_library(infer SHARED include/infer.hpp src/infer.cu)
add_library(cuda_tools SHARED include/cuda_tools.hpp src/cuda_tools.cu)
add_library(check SHARED include/check.hpp src/check.cpp)