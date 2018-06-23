#pragma once

#include <ogl.h>
#include "demo_loader.h"

class Scene;

// A special value to for "tex_type_count" in order to use all available textures.
#define ALL_TEXTURES 999

// Default clear color.
const float g_default_clear_color[] = { 0.0f, 0.0f, 0.0f, 1.0f };

// A renderer that takes the meshes in a given scene and renders them into the given framebuffer using the provided settings.
class SceneRenderer
{
public:
    SceneRenderer();
    ~SceneRenderer();
    void render(Scene* scene, 
				dw::Framebuffer* fbo, 
				dw::Program* global_shader, 
				uint32_t w, 
				uint32_t h, 
				uint32_t clear_flags = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT, 
				float* clear_color = (float*)g_default_clear_color, 
				uint32_t tex_type_count = ALL_TEXTURES, 
				uint32_t* tex_types = nullptr);
    
private:
    int32_t m_texture_flags[6] =
    {
        TEXTURE_ALBEDO,
        TEXTURE_NORMAL,
        TEXTURE_METALNESS,
		TEXTURE_ROUGHNESS,
        TEXTURE_DISPLACEMENT,
        TEXTURE_EMISSIVE
    };
    
    std::string m_texture_uniform_names[6] =
    {
        "s_Albedo",
        "s_Normal",
		"s_Metalness",
        "s_Roughness",
        "s_Displacement",
        "s_Emissive"
    };
    
    const uint32_t m_texture_count = 6;
};
