#pragma once

#include <stdint.h>
#include <string>

namespace nimble
{
struct Viewport
{
    std::string name;
    float       x_scale = 0.0f;
    float       y_scale = 0.0f;
    float       w_scale = 1.0f;
    float       h_scale = 1.0f;
    int32_t     z_order = 0;
};
} // namespace nimble