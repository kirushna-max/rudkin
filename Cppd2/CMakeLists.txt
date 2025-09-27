# Minimum CMake version
cmake_minimum_required(VERSION 3.10)

# Project name
project(MyOpenCVProject)

# Find the OpenCV package installed on your system
find_package(OpenCV REQUIRED)

# Add your executable
add_executable(circles src/circles.cpp)

# Link your program against the OpenCV libraries
target_link_libraries(circles ${OpenCV_LIBS})
