cmake_minimum_required(VERSION 2.8) 
project(gpair)

#set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "-g -D DEBUG -O0")
#set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -Wshadow")
#set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# Call CMake on the subdirectories
SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)
SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMakeModules)

# Include liboi and other requirements(textio, getoifits, oifitslib)
add_subdirectory(${CMAKE_SOURCE_DIR}/lib/textio)
include_directories(${CMAKE_SOURCE_DIR}/lib/textio/include)
add_subdirectory(${CMAKE_SOURCE_DIR}/lib/liboi)
include_directories(${CMAKE_SOURCE_DIR}/lib/liboi/src)
include_directories(${CMAKE_SOURCE_DIR}/lib/liboi/lib/getoifits)
include_directories(${CMAKE_SOURCE_DIR}/lib/liboi/lib/oifitslib)

add_subdirectory(src)

# Copy over kernel and shader sources:
file(GLOB KERNELS ${CMAKE_SOURCE_DIR}/lib/liboi/src/kernels/*.cl)
file(COPY ${KERNELS} DESTINATION ${EXECUTABLE_OUTPUT_PATH}/kernels/)