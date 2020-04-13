# Disable in-source builds to prevent source tree corruption.
if(" ${CMAKE_SOURCE_DIR}" STREQUAL " ${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR "
FATAL: In-source builds are not allowed.
       You should create a separate directory for build files.
")
endif()

cmake_minimum_required(VERSION 2.8)
project(pyDBoW3)

# Project version
#------------------------------------------------------------------------------
set(pyDBOW3_MAJOR_VERSION 1)
set(pyDBOW3_MINOR_VERSION 1)
set(pyDBOW3_PATCH_VERSION 1)
set(pyDBOW3_VERSION ${pyDBOW3_MAJOR_VERSION}.${pyDBOW3_MINOR_VERSION}.${pyDBOW3_PATCH_VERSION})
set(pyDBOW3_VERSION_STR ${pyDBOW3_VERSION})
set(pyDBOW3_VERSION_ID ${pyDBOW3_MAJOR_VERSION}${pyDBOW3_MINOR_VERSION}${pyDBOW3_PATCH_VERSION})


# Building options
#-------------------------------------------------------------------------------
option(BUILD_PYTHON3 "Build DBow3 Python3 binding instead of default Python2." OFF)
OPTION(BUILD_STATICALLY_LINKED 	"Set to OFF to link with opencv and DBoW3 dynamically" ON)

SET(Boost_DEBUG OFF)

if(BUILD_PYTHON3)
	set(PYTHON_VERSION 3)
else()
	set(PYTHON_VERSION 2)
endif()

message( STATUS "BUILD_PYTHON3 = ${BUILD_PYTHON3}")
message( STATUS "PYTHON_VERSION = ${PYTHON_VERSION}")


# CMake options
#-----------------------------------------------------------------------------------------------------------------------

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall  -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive -Wall -O3")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    add_definitions(-DCOMPILEDWITHC11)
    message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    add_definitions(-DCOMPILEDWITHC0X)
    message(STATUS "Using flag -std=c++0x.")
else()
    #message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

if (COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif (COMMAND cmake_policy)

# Add cmake_modules to module
set(CMAKE_MODULE_PATH
        ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules")

# Set the release mode if not specified
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

if (CMAKE_COMPILER_IS_GNUCXX)
    add_definitions("-fPIC")
endif ()


#-----------------------------------------------------------------------------------------------------------------------

#LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../../cmake_modules)

IF(NOT UNIX)
  SET(OpenCV_USE_STATIC_LIBS ON)
ENDIF()
find_package(OpenCV REQUIRED)

SET(DBoW3_USE_STATIC_LIBS ON)
find_package(DBoW3 REQUIRED)

#-----------------------------------------------------------------------------------------------------------------------
# Python binding
#-----------------------------------------------------------------------------------------------------------------------

if (BUILD_PYTHON3)
  FIND_PACKAGE(PythonInterp 3 REQUIRED)
  FIND_PACKAGE(PythonLibs 3 REQUIRED)
else()
  SET(PythonLibs_FIND_VERSION 2.7)
  FIND_PACKAGE(PythonInterp 2.7 REQUIRED)
  FIND_PACKAGE(PythonLibs 2.7 REQUIRED)
endif()

if(PYTHON_VERSION_STRING VERSION_LESS 2.7)
  message(WARNING "Python version less than 2.7: \"${PYTHON_VERSION_STRING}\".")
endif()

FIND_PACKAGE(NumPy REQUIRED)

IF (NOT UNIX)
  SET(Boost_USE_STATIC_LIBS ON)
  add_definitions(-DBOOST_PYTHON_STATIC_LIB)
ENDIF()
SET(Boost_USE_MULTITHREADED ON)
SET(Boost_USE_STATIC_RUNTIME OFF)
SET(BOOST_ALL_NO_LIB ON)

FIND_PACKAGE(Boost 1.54)

if (UNIX)
  FIND_PACKAGE(Boost 1.54 COMPONENTS python-py${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR} REQUIRED)
else()	
  FIND_PACKAGE(Boost COMPONENTS python${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR} REQUIRED)
endif()


# This is a hack, since DBoW3 report weird library path
if (UNIX)
  LINK_DIRECTORIES(${DBoW3_INCLUDE_DIRS}/../lib)
else()
  LINK_DIRECTORIES(../install/DBow3/build/bin/Release)
endif()

if (UNIX)
  INCLUDE_DIRECTORIES(
  ${OpenCV_INCLUDE_DIRS} 
  ${Boost_INCLUDE_DIRS} 
  ${PYTHON_INCLUDE_DIRS} 
  ${NUMPY_INCLUDES} 
  ${DBoW3_INCLUDE_DIRS}/DBoW3 
  lib/)
else()
  INCLUDE_DIRECTORIES(
  ${OpenCV_INCLUDE_DIRS} 
  ${Boost_INCLUDE_DIRS} 
  ${PYTHON_INCLUDE_DIRS} 
  ${NUMPY_INCLUDES} 
  ../install/DBow3/src
  lib/)
endif()

if (BUILD_STATICALLY_LINKED)
    ADD_DEFINITIONS(-DBOOST_PYTHON_STATIC_LIB)
endif ()

# Build np<=>opencv converter library
ADD_LIBRARY(np_opencv_converter lib/conversion.cpp lib/np_opencv_converter.cpp)
TARGET_COMPILE_OPTIONS(np_opencv_converter PRIVATE -DBOOST_ALL_NO_LIB)
if (UNIX)
  TARGET_LINK_LIBRARIES(np_opencv_converter ${OpenCV_LIBRARIES} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
else()
endif()


# Build python extension 
ADD_LIBRARY(pyDBoW3 SHARED dbow3.cpp)
TARGET_COMPILE_OPTIONS(pyDBoW3 PRIVATE -DBOOST_ALL_NO_LIB)

if (NOT UNIX)
  TARGET_LINK_LIBRARIES(pyDBoW3 np_opencv_converter ${OpenCV_LIBRARIES} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${DBoW3_LIBRARIES} "-static")
else ()
  TARGET_LINK_LIBRARIES(pyDBoW3 np_opencv_converter ${DBoW3_LIBRARIES})
endif ()

if (NOT UNIX)
  SET_TARGET_PROPERTIES(pyDBoW3 PROPERTIES PREFIX "" SUFFIX ".pyd")
else ()
  SET_TARGET_PROPERTIES(pyDBoW3 PROPERTIES PREFIX "")
endif()

# ----------------------------------------------------------------------------
# display status message for important variables
# ----------------------------------------------------------------------------
message( STATUS )
MESSAGE( STATUS "-------------------------------------------------------------------------------" )
message( STATUS "General configuration for ${PROJECT_NAME} ${PROJECT_VERSION}")
MESSAGE( STATUS "-------------------------------------------------------------------------------" )
message("    Built as static libs?:" ${BUILD_STATICALLY_LINKED})
message("    Compiler:"                   "${CMAKE_COMPILER}"   "${CMAKE_CXX_COMPILER}")

message( STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message( STATUS "C++ flags (Release):       ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
message( STATUS "C++ flags (Debug):         ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
message( STATUS "C++ flags (Relase+Debug):         ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")

message( STATUS "CMAKE_CXX_FLAGS:         ${CMAKE_CXX_FLAGS}")
message( STATUS "CMAKE_BINARY_DIR:         ${CMAKE_BINARY_DIR}")
message(STATUS "PYTHON_EXECUTABLE = ${PYTHON_EXECUTABLE}")
message(STATUS "PYTHON_LIBRARIES = ${PYTHON_LIBRARIES}")
message(STATUS "PYTHON_INCLUDE_DIRS = ${PYTHON_INCLUDE_DIRS}")
message(STATUS "NUMPY_LIBRARIES = ${NUMPY_LIBRARIES}")
message(STATUS "NUMPY_INCLUDE_DIRS = ${NUMPY_INCLUDES}")
message(STATUS "DBoW3_INCLUDE_DIRS = ${DBoW3_INCLUDE_DIRS}")
message(STATUS "DBoW3_LIB_DIR = ${DBoW3_LIB_DIR}")
message(STATUS "DBoW3_LIBRARIES = ${DBoW3_LIBRARIES}")
message(STATUS "Boost_LIBRARIES = ${Boost_LIBRARIES}")
message(STATUS "PROJECT LIBRARIES = ${PROJECT_LIBRARIES}")
