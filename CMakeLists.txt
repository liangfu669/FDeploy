# 视觉算法cpp版
cmake_minimum_required(VERSION 3.1)
project(vision_cpp)

set(CMAKE_CXX_STANDARD 14)
set(OpenCV_DIR /home/soft/opencv/lib/cmake/opencv4)
#set(Eigen3_DIR /home/soft/Eigen/share/eigen3/cmake)

find_package(OpenCV REQUIRED)
link_libraries(${OpenCV_LIBS} ${OpenCV_LIBRARIES})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(CUDA REQUIRED)      # 包含CUDA相关依赖
include_directories(${CUDA_INCLUDE_DIRS})
link_libraries(${CUDA_CUDART_LIBRARY})

include_directories(/home/liangfuchu/soft/TensorRT-8.5.2.2/include)  # 包含tensorrt
link_directories(/home/liangfuchu/soft/TensorRT-8.5.2.2/lib)
link_libraries(nvinfer)

add_subdirectory(utils_algorithm)
add_subdirectory(detect_algorithm)
add_subdirectory(classification_algorithm)

add_executable(main main.cpp)

target_link_libraries(main ${PROJECT_SOURCE_DIR}/cmake-build-debug/detect_algorithm/libyolov5.so)


