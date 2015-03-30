cmake_minimum_required (VERSION 3.0)
project (neuroEvolution)

set (CMAKE_BUILD_TYPE debug)
# set (CMAKE_VERBOSE_MAKEFILE ON)

if(CMAKE_COMPILER_IS_GNUCXX)
    set( CMAKE_CXX_FLAGS   "-O -Wall" )
    add_definitions(-std=gnu++0x)
endif()

add_executable(neuroevolution main.cpp feed_forward_nn.cpp neuron.cpp)
