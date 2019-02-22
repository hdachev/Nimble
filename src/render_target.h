#pragma once

#include <memory>
#include "ogl.h"

namespace nimble
{
struct RenderTarget
{
    uint32_t                 id;
    float                    scale_w;
    float                    scale_h;
    uint32_t                 w;
    uint32_t                 h;
    GLenum                   target;
    GLenum                   internal_format;
    GLenum                   format;
    GLenum                   type;
    uint32_t                 num_samples;
    uint32_t                 array_size;
    uint32_t                 mip_levels;
	std::string				 forward_slot;
    std::shared_ptr<Texture> texture;

    RenderTarget();

    bool is_scaled();
};

struct RenderTargetView
{
    uint32_t                 face;
    uint32_t                 layer;
    uint32_t                 mip_level;
    std::shared_ptr<Texture> texture;

    RenderTargetView()
    {
        face      = 0;
        layer     = 0;
        mip_level = 0;
        texture   = nullptr;
    }

    RenderTargetView(uint32_t _face, uint32_t _layer, uint32_t _mip_level, std::shared_ptr<Texture> _texture)
    {
        face      = (_texture->target() == GL_TEXTURE_CUBE_MAP || _texture->target() == GL_TEXTURE_CUBE_MAP_ARRAY) ? _face : 0;
        layer     = _layer;
        mip_level = _mip_level;
        texture   = _texture;
    }
};
} // namespace nimble
