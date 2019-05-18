#pragma once

#include <glm.hpp>
#include "macros.h"
#include "constants.h"

namespace nimble
{
struct PointLightData
{
    NIMBLE_ALIGNED(16)
    glm::vec4 position_range;
    NIMBLE_ALIGNED(16)
    glm::vec4 color_intensity;
    NIMBLE_ALIGNED(16)
    int32_t casts_shadow;
};

struct SpotLightData
{
    NIMBLE_ALIGNED(16)
    glm::vec4 position_cone_angle;
    NIMBLE_ALIGNED(16)
    glm::vec4 direction_range;
    NIMBLE_ALIGNED(16)
    glm::vec4 color_intensity;
    NIMBLE_ALIGNED(16)
    int32_t casts_shadow;
};

struct DirectionalLightData
{
    NIMBLE_ALIGNED(16)
    glm::vec4 direction;
    NIMBLE_ALIGNED(16)
    glm::vec4 color_intensity;
    NIMBLE_ALIGNED(16)
    int32_t casts_shadow;
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
    glm::mat4 spot_light_shadow_matrix[MAX_SHADOW_CASTING_SPOT_LIGHTS];
    glm::vec4 shadow_map_bias[MAX_POINT_LIGHTS]; // x = directional, y = spot, z = point
    glm::vec4 point_light_position_range[MAX_POINT_LIGHTS];
    glm::vec4 point_light_color_intensity[MAX_POINT_LIGHTS];
    glm::vec4 spot_light_position[MAX_SPOT_LIGHTS];
    glm::vec4 spot_light_cutoff_inner_outer[MAX_SPOT_LIGHTS];
    glm::vec4 spot_light_direction_range[MAX_SPOT_LIGHTS];
    glm::vec4 spot_light_color_intensity[MAX_SPOT_LIGHTS];
    glm::vec4 directional_light_direction[MAX_DIRECTIONAL_LIGHTS];
    glm::vec4 directional_light_color_intensity[MAX_DIRECTIONAL_LIGHTS];
    int32_t   point_light_casts_shadow[MAX_POINT_LIGHTS];
    int32_t   spot_light_casts_shadow[MAX_SPOT_LIGHTS];
    int32_t   directional_light_casts_shadow[MAX_DIRECTIONAL_LIGHTS];
    int32_t   point_light_count;
    int32_t   spot_light_count;
    int32_t   directional_light_count;
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