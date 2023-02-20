project(detect_algorithm)

include_directories(include)

add_library(yolov5 SHARED include/yolov5.h src/yolov5.cpp)