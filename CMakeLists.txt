# Copyright: (C) 2014 iCub Facility - Istituto Italiano di Tecnologia
# Authors: Giulia Pasquale
# CopyPolicy: Released under the terms of the GNU GPL v2.0.

cmake_minimum_required(VERSION 2.8.8)

set(PROJECTNAME caffeVSgie_feat_extraction)
project(${PROJECTNAME})

set(exe_gie   ${PROJECTNAME}_gie})
set(exe_caffe ${PROJECTNAME}_caffe)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

option(BUILD_GIE "Build GIE" ON)
option(BUILD_caffe "Build caffe" ON)

# OpenCV
find_package(OpenCV REQUIRED)
# CUDA
find_package(CUDA REQUIRED)

# CUDA compilation flags
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_53,code=sm_53)
list(APPEND CUDA_NVCC_FLAGS -O3)


if (BUILD_GIE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")	# -std=gnu++11

    set(folder_source_gie  main_GIE.cpp GIEFeatExtractor.cpp)
    set(folder_header_gie  GIEFeatExtractor.h cudaUtility.h)

    cuda_add_executable(${PROJECTNAME}_gie   ${folder_source_gie}   ${folder_header_gie})

    target_link_libraries(${PROJECTNAME}_gie ${OpenCV_LIBRARIES})
    target_link_libraries(${PROJECTNAME}_gie nvcaffe_parser nvinfer)


endif(BUILD_GIE)

if (BUILD_caffe)

  find_package(Caffe)
  find_package(Boost)

  if(NOT Caffe_FOUND)
    message("Caffe NOT found!")
  else()
    message("Caffe found!")
  endif()

  include_directories(${Caffe_INCLUDE_DIRS})
  include_directories(${Boost_INCLUDE_DIRS})

  add_definitions(${Caffe_DEFINITIONS})    # ex. -DCPU_ONLY
  set(folder_source_caffe main_caffe.cpp)
  set(folder_header_caffe CaffeFeatExtractor.hpp)

  cuda_add_executable(${PROJECTNAME}_caffe ${folder_source_caffe} ${folder_header_caffe})
  target_link_libraries(${PROJECTNAME}_caffe ${Caffe_LIBRARIES})
  target_link_libraries(${PROJECTNAME}_caffe ${Boost_LIBRARIES})
  target_link_libraries(${PROJECTNAME}_caffe ${OpenCV_LIBRARIES})

endif(BUILD_caffe)
