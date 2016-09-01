# Copyright: (C) 2014 iCub Facility - Istituto Italiano di Tecnologia
# Authors: Giulia Pasquale
# CopyPolicy: Released under the terms of the GNU GPL v2.0.

cmake_minimum_required(VERSION 2.8.8)

set(DL_LIB "gie" CACHE FILEPATH "gie or caffe")

set(PROJECTNAME caffeVSgie_feat_extraction)
project(${PROJECTNAME})

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

if (${DL_LIB} STREQUAL "gie")
    set(GIE_PATH "$ENV{ROBOT_CODE}/GIE" CACHE FILEPATH "Path to GPU Inference Engine")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")	# -std=gnu++11
elseif (${DL_LIB} STREQUAL "caffe")
    find_package(Caffe REQUIRED)
endif()

# OpenCV
find_package(OpenCV REQUIRED)
# CUDA
find_package(CUDA REQUIRED)

if (${DL_LIB} STREQUAL "caffe")
    find_package(Boost)
endif()

# CUDA compilation flags
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_53,code=sm_53)
list(APPEND CUDA_NVCC_FLAGS -std=c++11 -O3)

if (${DL_LIB} STREQUAL "gie")
    include_directories(${GIE_PATH}/include)
    link_directories(${GIE_PATH}/lib)
elseif (${DL_LIB} STREQUAL "caffe")
    include_directories(${Caffe_INCLUDE_DIRS})
    include_directories(${Boost_INCLUDE_DIRS}) 
endif()

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})

if (${DL_LIB} STREQUAL "gie")
    set(folder_source main_GIE.cpp GIEFeatExtractor.cpp)
    set(folder_header GIEFeatExtractor.h cudaUtility.h)
elseif (${DL_LIB} STREQUAL "caffe")
    add_definitions(${Caffe_DEFINITIONS})    # ex. -DCPU_ONLY
    set(folder_source main_caffe.cpp)
    set(folder_header CaffeFeatExtractor.hpp)
endif()
source_group("Source Files" FILES ${folder_source})
source_group("Header Files" FILES ${folder_header})

cuda_add_executable(${PROJECTNAME} ${folder_source} ${folder_header})

if (${DL_LIB} STREQUAL "gie")
    target_link_libraries(${PROJECTNAME} nvcaffe_parser nvinfer)
elseif (${DL_LIB} STREQUAL "caffe")
    target_link_libraries(${PROJECTNAME} ${Caffe_LIBRARIES})
    target_link_libraries(${PROJECTNAME} ${Boost_LIBRARIES})
endif()

target_link_libraries(${PROJECTNAME} ${OpenCV_LIBRARIES})
target_link_libraries(${PROJECTNAME} ${CUDA_LIBRARIES} )

