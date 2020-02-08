#pragma once

#include <glm.hpp>
#include "macros.h"
#include "constants.h"

namespace nimble
{
struct LightData
{
    // | Spot                     | Directional                                        | Point                    |
    glm::ivec4 indices0; // | x: shadow map, y: matrix | x: first shadow map index, y: first cascade matrix | x: shadow map            |
    glm::ivec4 indices1; // |                          |                                                    |                          |
    glm::vec4  data0;    // | xyz: position, w: bias   | xyz: direction, w: bias                            | xyz: positon, w: bias    |
    glm::vec4  data1;    // | xy: cutoff               | xyz: color, w: intensity                           | x: near, y: far          |
    glm::vec4  data2;    // | xyz: direction, w: range | xyzw: far planes                                   | xyz: color, w: intensity |
    glm::vec4  data3;    // | xyz: color, w: intensity |                                                    |                          |
};

struct PerViewUniforms
{
    glm::mat4 last_view_proj;
    glm::mat4 view_proj;
    glm::mat4 inv_view_proj;
    glm::mat4 inv_proj;
    glm::mat4 inv_view;
    glm::mat4 proj_mat;
    glm::mat4 view_mat;
    glm::mat4 cascade_matrix[MAX_SHADOW_CASTING_DIRECTIONAL_LIGHTS * MAX_SHADOW_MAP_CASCADES];
    glm::vec4 view_pos;
    glm::vec4 view_dir;
    glm::vec4 current_prev_jitter;
    glm::vec4 z_buffer_params;
    glm::vec4 time_params;
    glm::vec4 viewport_params;
    float     cascade_far_plane[MAX_SHADOW_CASTING_DIRECTIONAL_LIGHTS * MAX_SHADOW_MAP_CASCADES];
    float     tan_half_fov;
    float     aspect_ratio;
    float     near_plane;
    float     far_plane;
    int32_t   num_cascades;
    int32_t   viewport_width;
    int32_t   viewport_height;
    uint8_t   padding[4];
};

struct PerEntityUniforms
{
    NIMBLE_ALIGNED(16)
    glm::mat4 modal_mat;
    NIMBLE_ALIGNED(16)
    glm::mat4 last_model_mat;
    uint8_t   padding[128];
};

struct PerSceneUniforms
{
    LightData  lights[MAX_LIGHTS];
    glm::mat4  shadow_matrices[MAX_SHADOW_CASTING_SPOT_LIGHTS + MAX_SHADOW_CASTING_DIRECTIONAL_LIGHTS * MAX_SHADOW_MAP_CASCADES];
    glm::uvec4 light_count;
};

struct PerMaterialUniforms
{
    NIMBLE_ALIGNED(16)
    glm::vec4 albedo;
    NIMBLE_ALIGNED(16)
    glm::vec4 emissive;
    NIMBLE_ALIGNED(16)
    glm::vec4 metalness_roughness;
};
} // namespace nimble