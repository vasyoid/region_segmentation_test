cmake_minimum_required(VERSION 3.9)
project(region_segmentation_test)

set(CMAKE_CXX_STANDARD 11)

include_directories(include)
include_directories(bk_maxflow)

add_executable(region_segmentation_test
        src/main.cpp
        src/utils.cpp
        bk_maxflow/graph.cpp
        bk_maxflow/maxflow.cpp
        bk_maxflow/instances.inc)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(region_segmentation_test ${OpenCV_LIBS})
