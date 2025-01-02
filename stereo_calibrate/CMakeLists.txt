CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
PROJECT(main)
ENABLE_LANGUAGE(CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_DEBUG "-ggdb3 -O0 -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-g -O3 -Wall")

# Find OpenCV
FIND_PACKAGE(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Add executable
add_executable(stereo_calibrate main.cpp)

# Link OpenCV libraries
target_link_libraries(stereo_calibrate ${OpenCV_LIBS})
