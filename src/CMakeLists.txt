cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

file(GLOB_RECURSE NIMBLE_HEADERS ${PROJECT_SOURCE_DIR}/src/*.h)

file(GLOB_RECURSE NIMBLE_SOURCE ${PROJECT_SOURCE_DIR}/src/*.cpp)

list(APPEND NIMBLE_SOURCE ${PROJECT_SOURCE_DIR}/external/imgui/imgui.cpp
                          ${PROJECT_SOURCE_DIR}/external/imgui/imgui_demo.cpp
                          ${PROJECT_SOURCE_DIR}/external/imgui/imgui_draw.cpp
                          ${PROJECT_SOURCE_DIR}/external/imgui/imgui_widgets.cpp
                          ${PROJECT_SOURCE_DIR}/external/ImGuizmo/ImGuizmo.cpp)

list(APPEND NIMBLE_SOURCE ${PROJECT_SOURCE_DIR}/src/glad.c)
list(APPEND NIMBLE_SOURCE ${PROJECT_SOURCE_DIR}/src/external/nfd/nfd_common.c)

if (APPLE)
    list(APPEND NIMBLE_SOURCE ${PROJECT_SOURCE_DIR}/src/external/nfd/nfd_cocoa.m)
elseif (WIN32)
    list(APPEND NIMBLE_SOURCE ${PROJECT_SOURCE_DIR}/src/external/nfd/nfd_win.cpp)
elseif (LINUX)
    list(APPEND NIMBLE_SOURCE ${PROJECT_SOURCE_DIR}/src/external/nfd/nfd_gtk.c)
endif()

if (APPLE)
    add_executable(Nimble MACOSX_BUNDLE ${NIMBLE_HEADERS} ${NIMBLE_SOURCE})				
else()
    add_executable(Nimble ${NIMBLE_HEADERS} ${NIMBLE_SOURCE})
endif()

target_link_libraries(Nimble AssetCoreRuntime)
target_link_libraries(Nimble glfw)

if (APPLE)
    add_custom_command(TARGET Nimble POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/src/shader $<TARGET_FILE_DIR:Nimble>/Nimble.app/Contents/Resources/assets/shader)
else()
    add_custom_command(TARGET Nimble POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/src/shader $<TARGET_FILE_DIR:Nimble>/assets/shader)
endif()
