# Copyright (c) 2017 - 2024 Advanced Micro Devices, Inc. All rights reserved.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#  
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#  
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

cmake_minimum_required(VERSION 3.10)

set(VERSION "1.0.1")
project(vx_amd_custom VERSION ${VERSION} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)

# ROCM Path
if(DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Default ROCm installation path")
elseif(ROCM_PATH)
    message("-- ${PROJECT_NAME}: INFO -- ROCM_PATH Set -- ${ROCM_PATH}")
else()
    set(ROCM_PATH /opt/rocm CACHE PATH "Default ROCm installation path")
endif()

list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/hip)

# Find Packages
# HIP
if(GPU_SUPPORT AND "${BACKEND}" STREQUAL "HIP")
    if(NOT DEFINED HIP_PATH)
        if(NOT DEFINED ENV{HIP_PATH})
            set(HIP_PATH ${ROCM_PATH} CACHE PATH "Path to which HIP has been installed")
        else()
            set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
        endif()
    endif()
    list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/hip)
    find_package(HIP QUIET)
    if(NOT HIP_FOUND)
        message("-- ${Red}${PROJECT_NAME} WARNING: HIP Not Found${ColourReset}")
    endif()
endif()
# OpenMP
find_package(OpenMP QUIET)
if(OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    include_directories(${OpenMP_INCLUDE_DIRS})
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} OpenMP::OpenMP_CXX)
    message("-- ${White}${PROJECT_NAME} -- OpenMP Found${ColourReset}")
else()
    message("-- ${Yellow}WARNING: ${PROJECT_NAME} -- OpenMP Not FOUND${ColourReset}")
endif()

list(APPEND SOURCES source/custom_kernels.cpp source/custom_layer.cpp)
include_directories(../../amd_openvx/openvx/include ${ROCM_PATH}/include include ./custom_lib)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2")

if(HIP_FOUND AND OpenMP_FOUND)
    message("-- ${Green}${PROJECT_NAME} -- Building HIP backend${ColourReset}")
    set(ENABLE_HIP 1)
    link_directories(${HIP_PATH}/lib)
    list(APPEND PACKAGE_DEPENDS PACKAGE HIP)
    add_subdirectory(custom_lib)
    add_library(${PROJECT_NAME} SHARED ${SOURCES} $<TARGET_OBJECTS:custom_lib>)
    target_compile_definitions(${PROJECT_NAME} PRIVATE __HIP_PLATFORM_AMD__)
    target_link_libraries(${PROJECT_NAME} openvx hip::host ${LINK_LIBRARY_LIST})
    target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_HIP=${ENABLE_HIP})
    set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})
    # install MIVisionX libs -- {ROCM_PATH}/lib -- TBD: Remove duplicate install 
    install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT runtime NAMELINK_SKIP)
    install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT dev NAMELINK_ONLY)
    install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT asan)
    # install MIVisionX include files -- {ROCM_PATH}/include/mivisionx/
    install(FILES include/vx_amd_custom.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mivisionx COMPONENT dev)
else()
    message("-- ${Red}WARNING:${PROJECT_NAME} -- CPU backend not supported${ColourReset}")
endif()
