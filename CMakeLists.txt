cmake_minimum_required(VERSION 3.5)
project(SQPEngine)
add_library (SQPEngine 
            sqp_engine/sqp_engine.h 
            sqp_engine/sqp_engine.cpp)
target_compile_options(SQPEngine PRIVATE -Wall)

set(INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} )


find_package(Eigen3 REQUIRED)
list(APPEND INCLUDE_DIRS 
     ${EIGEN3_INCLUDE_DIR} 
     ${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${INCLUDE_DIRS} )

# when other libraries or executables link to SQPEngine
target_include_directories (SQPEngine PUBLIC ${INCLUDE_DIRS})
