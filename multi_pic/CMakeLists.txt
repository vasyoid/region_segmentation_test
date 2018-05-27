cmake_minimum_required(VERSION 3.9)
project(multi_pic)

set(CMAKE_CXX_STANDARD 11)

include_directories(include)
include_directories(bk_maxflow)
include_directories(edison_gpu/src, edison_gpu/segm)

add_subdirectory(edison_gpu)

file(GLOB incl_dir "include/*")
file(GLOB bk_dir "bk_maxflow/*")

set(SHIFT_LIB edison_gpu)
set(PLATFORM_SPECIFIC_LIBS "-lpthread")

add_executable(multi_pic
        src/grabcut.cpp
        src/cut.cpp
        src/magic_wand.cpp
        src/main.cpp
        src/shift.cpp
        ${bk_dir}
        ${incl_dir})

add_executable(test_wand
        src/test_wand.cpp
        ${shift_dir}
        ${shift_segm_dir}
        ${incl_dir})

add_executable(test_shift
        src/shift.cpp
        ${incl_dir})

add_executable(test_hist
        src/hist.cpp)

find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(multi_pic ${OpenCV_LIBS} ${SHIFT_LIB} ${PLATFORM_SPECIFIC_LIBS})
target_link_libraries(test_wand ${OpenCV_LIBS})
target_link_libraries(test_hist ${OpenCV_LIBS})
