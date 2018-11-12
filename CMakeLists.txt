cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project("Nimble")

IF(APPLE)
	set(CMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LANGUAGE_STANDARD "c++14")
	set(CMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LIBRARY "libc++")
ENDIF()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

set(JSON_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/external/json/single_include/nlohmann")
set(ASSET_CORE_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/external/AssetCore/include")

add_subdirectory(external/dwSampleFramework)
add_subdirectory(external/AssetCore)

include_directories("${DW_SAMPLE_FRAMEWORK_INCLUDES}" "${JSON_INCLUDE_DIR}" "${ASSET_CORE_INCLUDE_DIR}")

add_subdirectory(src)