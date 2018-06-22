cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(NIMBLE_HEADERS ${PROJECT_SOURCE_DIR}/src/demo_loader.h
                   ${PROJECT_SOURCE_DIR}/src/entity.h
                   ${PROJECT_SOURCE_DIR}/src/renderer.h
                   ${PROJECT_SOURCE_DIR}/src/scene_renderer.h
                   ${PROJECT_SOURCE_DIR}/src/forward_renderer.h
                   ${PROJECT_SOURCE_DIR}/src/shadow_map_renderer.h
                   ${PROJECT_SOURCE_DIR}/src/scene.h)

set(NIMBLE_SOURCE ${PROJECT_SOURCE_DIR}/src/main.cpp
                  ${PROJECT_SOURCE_DIR}/src/demo_loader.cpp
                  ${PROJECT_SOURCE_DIR}/src/renderer.cpp
                  ${PROJECT_SOURCE_DIR}/src/scene_renderer.cpp
                  ${PROJECT_SOURCE_DIR}/src/forward_renderer.cpp
                  ${PROJECT_SOURCE_DIR}/src/shadow_map_renderer.cpp
                  ${PROJECT_SOURCE_DIR}/src/scene.cpp)
                         
add_executable(Nimble ${NIMBLE_HEADERS} ${NIMBLE_SOURCE})				

target_link_libraries(Nimble dwSampleFramework)