project(detect_algorithm)

include_directories(include)

add_library(yolov5 SHARED include/yolov5.h src/yolov5.cpp)

add_library(yolo SHARED include/yolo.hpp src/yolo.cu)
add_library(infer SHARED include/infer.hpp src/infer.cu)