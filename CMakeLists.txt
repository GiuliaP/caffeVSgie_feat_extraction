# Copyright: (C) 2014 iCub Facility - Istituto Italiano di Tecnologia
# Authors: Giulia Pasquale
# CopyPolicy: Released under the terms of the GNU GPL v2.0.

cmake_minimum_required(VERSION 2.8.8)

set(PROJECTNAME caffe_feat_extraction)
project(${PROJECTNAME})

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

# GIE
set(GIE_PATH "$ENV{ROBOT_CODE}/GIE" CACHE FILEPATH "Path to GPU Inference Engine")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")	# -std=gnu++11
# Caffe
find_package(Caffe REQUIRED)

# OpenCV
find_package(OpenCV REQUIRED)
# CUDA
find_package(CUDA REQUIRED)
# Boost
set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost COMPONENTS filesystem)

# CUDA compilation flags
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_53,code=sm_53)
list(APPEND CUDA_NVCC_FLAGS -std=c++11 -O3)

# GIE
include_directories(${GIE_PATH}/include)
link_directories(${GIE_PATH}/lib)
# Caffe
include_directories(${Caffe_INCLUDE_DIRS})

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS}) 

include_directories(${PROJECT_SOURCE_DIR})

add_definitions(${Caffe_DEFINITIONS})    # ex. -DCPU_ONLY

set(folder_source main_GIEvsCaffe.cpp GIEFeatExtractor.cpp)
set(folder_header CaffeFeatExtractor.hpp GIEFeatExtractor.h cudaUtility.h)
source_group("Source Files" FILES ${folder_source})
source_group("Header Files" FILES ${folder_header})

cuda_add_executable(${PROJECTNAME} ${folder_source} ${folder_header})

# Caffe
target_link_libraries(${PROJECTNAME} ${Caffe_LIBRARIES})
# GIE
target_link_libraries(${PROJECTNAME} nvcaffe_parser nvinfer)

target_link_libraries(${PROJECTNAME} ${OpenCV_LIBRARIES})
target_link_libraries(${PROJECTNAME} ${Boost_LIBRARIES})

target_link_libraries(${PROJECTNAME} ${CUDA_LIBRARIES} )

