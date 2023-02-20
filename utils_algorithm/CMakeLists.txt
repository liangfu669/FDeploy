project(utils_algorithm)

include_directories(include)

add_library(color_extraction SHARED include/color_extraction.h src/color_extraction.cpp)
add_library(match_rectification SHARED include/match_rectification.h src/match_rectification.cpp)
add_library(enhancement SHARED include/enhancement.h src/enhancement.cpp)