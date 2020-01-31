#pragma once

#include <stdint.h>
#include "transform.h"

namespace nimble
{
enum LightType
{
    LIGHT_TYPE_DIRECTIONAL = 0,
    LIGHT_TYPE_SPOT,
    LIGHT_TYPE_POINT
};

struct Light
{
    bool      enabled;
    bool      casts_shadow;
    float     shadow_map_bias;
    glm::vec3 color;
    float     intensity;
    Transform transform;
};

struct DirectionalLight : public Light
{
    using ID = uint32_t;

    ID id;
};

struct PointLight : public Light
{
    using ID = uint32_t;

    ID    id;
    float range;
};

struct SpotLight : public Light
{
    using ID = uint32_t;

    ID    id;
    float range;
    float inner_cone_angle;
    float outer_cone_angle;
};
} // namespace nimble