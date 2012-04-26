cmake_minimum_required(VERSION 2.8) 
project(gpair CXX)

# include directories for the oilib:
include_directories(${OI_INCLUDE_DIRS})

if (CMAKE_BUILD_TOOL MATCHES "make")
    #add_definitions(-Wall  -Wno-deprecated -Werror -Wshadow )
endif (CMAKE_BUILD_TOOL MATCHES "make")

# Assemble all of the source to build gpair.
file(GLOB SOURCE *.cpp)
add_executable(gpair ${SOURCE})
SET_TARGET_PROPERTIES(gpair PROPERTIES LINKER_LANGUAGE Fortran)
target_link_libraries(gpair textio oi)
