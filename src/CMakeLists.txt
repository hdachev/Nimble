cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(NIMBLE_HEADERS ${PROJECT_SOURCE_DIR}/src/demo_loader.h
                   ${PROJECT_SOURCE_DIR}/src/entity.h
                   ${PROJECT_SOURCE_DIR}/src/renderer.h
                   ${PROJECT_SOURCE_DIR}/src/scene_renderer.h
                   ${PROJECT_SOURCE_DIR}/src/forward_renderer.h
                   ${PROJECT_SOURCE_DIR}/src/g_buffer_renderer.h
                   ${PROJECT_SOURCE_DIR}/src/shadow_map_renderer.h
                   ${PROJECT_SOURCE_DIR}/src/deferred_shading_renderer.h
                   ${PROJECT_SOURCE_DIR}/src/scene.h
                   ${PROJECT_SOURCE_DIR}/src/constants.h
                   ${PROJECT_SOURCE_DIR}/src/uniforms.h
                   ${PROJECT_SOURCE_DIR}/src/final_composition.h
                   ${PROJECT_SOURCE_DIR}/src/global_graphics_resources.h
                   ${PROJECT_SOURCE_DIR}/src/post_process_renderer.h
                   ${PROJECT_SOURCE_DIR}/src/uniforms.h
                   ${PROJECT_SOURCE_DIR}/src/csm.h
                   ${PROJECT_SOURCE_DIR}/src/external/nfd/nfd.h
                   ${PROJECT_SOURCE_DIR}/src/external/nfd/nfd_common.h
                   ${PROJECT_SOURCE_DIR}/src/external/nfd/common.h)

set(NIMBLE_SOURCE ${PROJECT_SOURCE_DIR}/src/main.cpp
                  ${PROJECT_SOURCE_DIR}/src/demo_loader.cpp
                  ${PROJECT_SOURCE_DIR}/src/renderer.cpp
                  ${PROJECT_SOURCE_DIR}/src/scene_renderer.cpp
                  ${PROJECT_SOURCE_DIR}/src/forward_renderer.cpp
                  ${PROJECT_SOURCE_DIR}/src/g_buffer_renderer.cpp
                  ${PROJECT_SOURCE_DIR}/src/deferred_shading_renderer.cpp
                  ${PROJECT_SOURCE_DIR}/src/shadow_map_renderer.cpp
                  ${PROJECT_SOURCE_DIR}/src/scene.cpp
                  ${PROJECT_SOURCE_DIR}/src/csm.cpp
                  ${PROJECT_SOURCE_DIR}/src/final_composition.cpp
                  ${PROJECT_SOURCE_DIR}/src/global_graphics_resources.cpp
                  ${PROJECT_SOURCE_DIR}/src/post_process_renderer.cpp
                  ${PROJECT_SOURCE_DIR}/src/external/nfd/nfd_common.c)

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

target_link_libraries(Nimble dwSampleFramework)

if (APPLE)
    add_custom_command(TARGET Nimble POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/src/shader $<TARGET_FILE_DIR:Nimble>/Nimble.app/Contents/Resources/assets/shader)
else()
    add_custom_command(TARGET Nimble POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/src/shader $<TARGET_FILE_DIR:Nimble>/assets/shader)
endif()
