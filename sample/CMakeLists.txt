project(sample)

add_executable(yolov5_sample yolov5_sample.cpp)
add_executable(hogSvm_sample hogSvm_sample.cpp)
add_executable(Retinex_sample Retinex_sample.cpp)

target_link_libraries(yolov5_sample ${CMAKE_BINARY_DIR}/detect_algorithm/libyolov5.so)
target_link_libraries(hogSvm_sample ${CMAKE_BINARY_DIR}/classification_algorithm/libhogSvm.so)
target_link_libraries(Retinex_sample ${CMAKE_BINARY_DIR}/utils_algorithm/libenhancement.so)