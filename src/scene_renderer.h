#pragma once

#include <ogl.h>
#include "demo_loader.h"

class Scene;

class SceneRenderer
{
public:
    SceneRenderer();
    ~SceneRenderer();
    void render(Scene* scene, dw::Framebuffer* fbo, uint32_t w, uint32_t h, uint32_t clear_flags, float* clear_color, uint32_t flags);
    
private:
    const uint32_t m_texture_flags[6] =
    {
        TEXTURE_ALBEDO,
        TEXTURE_NORMAL,
        TEXTURE_ROUGHNESS,
        TEXTURE_METALNESS,
        TEXTURE_DISPLACEMENT,
        TEXTURE_EMISSIVE
    };
    
    const char* m_texture_uniform_names[6] =
    {
        "s_Albedo",
        "s_Normal",
        "s_Roughness",
        "s_Metalness",
        "s_Displacement",
        "s_Emissive"
    };
    
    const uint32_t m_texture_count = 6;
};
