project(classification_algorithm)

add_library(hogSvm SHARED include/hogSvm.h src/hogSvm.cpp)
add_library(resnet SHARED include/resnet.h src/resnet.cpp)