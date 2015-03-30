cmake_minimum_required (VERSION 2.8)
project (neuroEvolution)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

set (CMAKE_BUILD_TYPE debug)
#set (CMAKE_VERBOSE_MAKEFILE ON)

if(CMAKE_COMPILER_IS_GNUCXX)
    set( CMAKE_CXX_FLAGS "-O2 -Wall -Wno-sign-compare" )
    add_definitions(-std=gnu++0x)
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)

add_library(neuroev src/main.cpp src/feed_forward_nn.cpp src/neuron.cpp src/functions.cpp)
add_executable(neuroEvolution.exe src/main.cpp src/feed_forward_nn.cpp src/neuron.cpp src/functions.cpp)

if(MAKE_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)
    include_directories(${GTEST_INCLUDE_DIRS})
    add_executable(test_neuroEvolution tests/test_neuroEvolution.cpp)
    add_test(AllTests bin/test_neuroEvolution)
    target_link_libraries(test_neuroEvolution ${GTEST_BOTH_LIBRARIES} neuroev)
endif()
