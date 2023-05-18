project(sample)



add_executable(hogSvm_sample hogSvm_sample.cpp)
add_executable(Retinex_sample Retinex_sample.cpp)
add_executable(yolo_sample yolo_sample.cpp)
add_executable(resnet_sample resnet_sample.cpp)
add_executable(unet_sample unet_sample.cpp)


target_link_libraries(resnet_sample ${CMAKE_BINARY_DIR}/algorithm/classify/libresnet.so
        /home/liangfuchu/code/cpp/vision_cpp/cmake-build-debug-cuda/algorithm/common/libcheck.so
        /home/liangfuchu/code/cpp/vision_cpp/cmake-build-debug-cuda/algorithm/common/libinfer.so
        /home/liangfuchu/code/cpp/vision_cpp/cmake-build-debug-cuda/algorithm/common/libcuda_tools.so)

target_link_libraries(yolo_sample ${CMAKE_BINARY_DIR}/algorithm/detect/libyolo.so
        /home/liangfuchu/code/cpp/vision_cpp/cmake-build-debug-cuda/algorithm/common/libcheck.so
        /home/liangfuchu/code/cpp/vision_cpp/cmake-build-debug-cuda/algorithm/common/libinfer.so
        /home/liangfuchu/code/cpp/vision_cpp/cmake-build-debug-cuda/algorithm/common/libcuda_tools.so)

target_link_libraries(unet_sample ${CMAKE_BINARY_DIR}/algorithm/segment/libunet.so
        /home/liangfuchu/code/cpp/vision_cpp/cmake-build-debug-cuda/algorithm/common/libcheck.so
        /home/liangfuchu/code/cpp/vision_cpp/cmake-build-debug-cuda/algorithm/common/libinfer.so
        /home/liangfuchu/code/cpp/vision_cpp/cmake-build-debug-cuda/algorithm/common/libcuda_tools.so)
