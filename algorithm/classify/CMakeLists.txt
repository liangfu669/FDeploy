project(classify)  # 项目名称
include_directories(../common)
add_library(resnet SHARED include/resnet.hpp src/resnet.cu)