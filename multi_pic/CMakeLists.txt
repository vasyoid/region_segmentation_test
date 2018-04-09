cmake_minimum_required(VERSION 3.9)
project(multi_pic)

set(CMAKE_CXX_STANDARD 11)

include_directories(include)
include_directories(bk_maxflow)

add_executable(multi_pic
        src/main.cpp
        src/grabcut.cpp
        src/utils.cpp
        src/cut.cpp
        bk_maxflow/graph.cpp
        bk_maxflow/maxflow.cpp
        bk_maxflow/instances.inc
        include/cut.h
        include/grabcut.h
        include/segment.h)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(multi_pic ${OpenCV_LIBS})
