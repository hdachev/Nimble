#pragma once

#include <stdint.h>
#include "transform.h"

namespace nimble
{
struct Light
{
    bool      enabled;
    bool      casts_shadow;
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
    float cone_angle;
};
} // namespace nimble