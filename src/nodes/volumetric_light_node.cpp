#include "volumetric_light_node.h"
#include "../render_graph.h"
#include "../resource_manager.h"
#include "../renderer.h"
#include "../logger.h"
#define _USE_MATH_DEFINES
#include <math.h>

namespace nimble
{
DEFINE_RENDER_NODE_FACTORY(VolumetricLightNode)

// -----------------------------------------------------------------------------------------------------------------------------------

VolumetricLightNode::VolumetricLightNode(RenderGraph* graph) :
    RenderNode(graph)
{
	m_flags = NODE_USAGE_PER_VIEW_UBO | NODE_USAGE_POINT_LIGHTS | NODE_USAGE_SPOT_LIGHTS | NODE_USAGE_DIRECTIONAL_LIGHTS | NODE_USAGE_SHADOW_MAPPING;
}

// -----------------------------------------------------------------------------------------------------------------------------------

VolumetricLightNode::~VolumetricLightNode()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void VolumetricLightNode::declare_connections()
{
    register_input_render_target("Color");
    register_input_render_target("Depth");

	m_color_rt = register_forwarded_output_render_target("Color");
	m_volumetric_light_rt = register_scaled_intermediate_render_target("VolumetricLight", 0.5f, 0.5f, GL_TEXTURE_2D, GL_RGB16F, GL_RGB, GL_HALF_FLOAT, 1, 1);
	m_blur_rt = register_scaled_intermediate_render_target("VolumetricLightBlur", 0.5f, 0.5f, GL_TEXTURE_2D, GL_RGB16F, GL_RGB, GL_HALF_FLOAT, 1, 1);
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool VolumetricLightNode::initialize(Renderer* renderer, ResourceManager* res_mgr)
{
	register_bool_parameter("Dither", m_dither);
	register_bool_parameter("Enabled", m_enabled);
	register_int_parameter("Num Samples", m_num_samples, 0, 32);
    register_float_parameter("Mie Scattering G", m_mie_g, 0.0f, 1.0f);

    m_depth_rt = find_input_render_target("Depth");

	std::vector<uint8_t> dither;

#ifdef DITHER_8_8
	int i = 0;
	int dither_size = 8;

	dither.resize(dither_size * dither_size);

	dither[i++] = (1.0f / 65.0f * 255); 
	dither[i++] = (49.0f / 65.0f * 255);
	dither[i++] = (13.0f / 65.0f * 255);
	dither[i++] = (61.0f / 65.0f * 255);
	dither[i++] = (4.0f / 65.0f * 255); 
	dither[i++] = (52.0f / 65.0f * 255);
	dither[i++] = (16.0f / 65.0f * 255);
	dither[i++] = (64.0f / 65.0f * 255);

	dither[i++] = (33.0f / 65.0f * 255);
	dither[i++] = (17.0f / 65.0f * 255);
	dither[i++] = (45.0f / 65.0f * 255);
	dither[i++] = (29.0f / 65.0f * 255);
	dither[i++] = (36.0f / 65.0f * 255);
	dither[i++] = (20.0f / 65.0f * 255);
	dither[i++] = (48.0f / 65.0f * 255);
	dither[i++] = (32.0f / 65.0f * 255);

	dither[i++] = (9.0f / 65.0f * 255); 
	dither[i++] = (57.0f / 65.0f * 255);
	dither[i++] = (5.0f / 65.0f * 255); 
	dither[i++] = (53.0f / 65.0f * 255);
	dither[i++] = (12.0f / 65.0f * 255);
	dither[i++] = (60.0f / 65.0f * 255);
	dither[i++] = (8.0f / 65.0f * 255); 
	dither[i++] = (56.0f / 65.0f * 255);

	dither[i++] = (41.0f / 65.0f * 255);
	dither[i++] = (25.0f / 65.0f * 255);
	dither[i++] = (37.0f / 65.0f * 255);
	dither[i++] = (21.0f / 65.0f * 255);
	dither[i++] = (44.0f / 65.0f * 255);
	dither[i++] = (28.0f / 65.0f * 255);
	dither[i++] = (40.0f / 65.0f * 255);
	dither[i++] = (24.0f / 65.0f * 255);

	dither[i++] = (3.0f / 65.0f * 255); 
	dither[i++] = (51.0f / 65.0f * 255);
	dither[i++] = (15.0f / 65.0f * 255);
	dither[i++] = (63.0f / 65.0f * 255);
	dither[i++] = (2.0f / 65.0f * 255); 
	dither[i++] = (50.0f / 65.0f * 255);
	dither[i++] = (14.0f / 65.0f * 255);
	dither[i++] = (62.0f / 65.0f * 255);

	dither[i++] = (35.0f / 65.0f * 255);
	dither[i++] = (19.0f / 65.0f * 255);
	dither[i++] = (47.0f / 65.0f * 255);
	dither[i++] = (31.0f / 65.0f * 255);
	dither[i++] = (34.0f / 65.0f * 255);
	dither[i++] = (18.0f / 65.0f * 255);
	dither[i++] = (46.0f / 65.0f * 255);
	dither[i++] = (30.0f / 65.0f * 255);

	dither[i++] = (11.0f / 65.0f * 255);
	dither[i++] = (59.0f / 65.0f * 255);
	dither[i++] = (7.0f / 65.0f * 255); 
	dither[i++] = (55.0f / 65.0f * 255);
	dither[i++] = (10.0f / 65.0f * 255);
	dither[i++] = (58.0f / 65.0f * 255);
	dither[i++] = (6.0f / 65.0f * 255); 
	dither[i++] = (54.0f / 65.0f * 255);

	dither[i++] = (43.0f / 65.0f * 255);
	dither[i++] = (27.0f / 65.0f * 255);
	dither[i++] = (39.0f / 65.0f * 255);
	dither[i++] = (23.0f / 65.0f * 255);
	dither[i++] = (42.0f / 65.0f * 255);
	dither[i++] = (26.0f / 65.0f * 255);
	dither[i++] = (38.0f / 65.0f * 255);
	dither[i++] = (22.0f / 65.0f * 255);
#else
	int i = 0;
	int dither_size = 4;

	dither.resize(dither_size * dither_size);
	
	dither[i++] = (0.0f / 16.0f * 255); 
    dither[i++] = (8.0f / 16.0f * 255); 
    dither[i++] = (2.0f / 16.0f * 255); 
    dither[i++] = (10.0f / 16.0f * 255);

    dither[i++] = (12.0f / 16.0f * 255);
    dither[i++] = (4.0f / 16.0f * 255); 
    dither[i++] = (14.0f / 16.0f * 255);
    dither[i++] = (6.0f / 16.0f * 255); 
		 
    dither[i++] = (3.0f / 16.0f * 255); 
    dither[i++] = (11.0f / 16.0f * 255);
    dither[i++] = (1.0f / 16.0f * 255); 
    dither[i++] = (9.0f / 16.0f * 255); 
		   
    dither[i++] = (15.0f / 16.0f * 255);
    dither[i++] = (7.0f / 16.0f * 255); 
    dither[i++] = (13.0f / 16.0f * 255);
    dither[i++] = (5.0f / 16.0f * 255); 
#endif

    m_dither_texture = std::make_unique<Texture2D>(dither_size, dither_size, 1, 1, 1, GL_R8, GL_RED, GL_UNSIGNED_BYTE);
    m_dither_texture->set_min_filter(GL_NEAREST);
    m_dither_texture->set_mag_filter(GL_NEAREST);
    m_dither_texture->set_wrapping(GL_REPEAT, GL_REPEAT, GL_REPEAT);
	m_dither_texture->set_data(0, 0, dither.data());

    m_volumetrics_rtv = RenderTargetView(0, 0, 0, m_color_rt->texture);

    m_fullscreen_triangle_vs = res_mgr->load_shader("shader/post_process/fullscreen_triangle_vs.glsl", GL_VERTEX_SHADER);
    m_volumetrics_fs = res_mgr->load_shader("shader/post_process/volumetric_light/volumetric_light_fs.glsl", GL_FRAGMENT_SHADER, m_flags, renderer);

    if (m_fullscreen_triangle_vs && m_volumetrics_fs)
    {
        m_volumetrics_program = renderer->create_program(m_fullscreen_triangle_vs, m_volumetrics_fs);
        return true;
    }
    else
        return false;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void VolumetricLightNode::execute(double delta, Renderer* renderer, Scene* scene, View* view)
{
    volumetrics(renderer, scene, view);
	blur(renderer, scene, view);
	upscale(renderer, scene, view);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void VolumetricLightNode::shutdown()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::string VolumetricLightNode::name()
{
    return "Volumetric Light";
}

// -----------------------------------------------------------------------------------------------------------------------------------

void VolumetricLightNode::volumetrics(Renderer* renderer, Scene* scene, View* view)
{
	if (m_enabled)
	{
		glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "Volumetric Light Buffer");

		glDisable(GL_DEPTH_TEST);
		glDisable(GL_CULL_FACE);

		glEnable(GL_BLEND);
		glBlendFunc(GL_ONE, GL_ONE);

		m_volumetrics_program->use();

		renderer->bind_render_targets(1, &m_volumetrics_rtv, nullptr);
		glViewport(0, 0, m_graph->window_width(), m_graph->window_height());

		int32_t tex_unit = 0;

		if (m_volumetrics_program->set_uniform("s_Depth", tex_unit))
			m_depth_rt->texture->bind(tex_unit++);

		if (m_volumetrics_program->set_uniform("s_Dither", tex_unit))
			m_dither_texture->bind(tex_unit++);

		float g_2 = m_mie_g * m_mie_g;
		m_volumetrics_program->set_uniform("u_MieG", glm::vec4(1.0f - g_2, 1.0f + g_2, 2.0f * m_mie_g, 1.0f / (4.0f * M_PI)));
		m_volumetrics_program->set_uniform("u_NumSamples", m_num_samples);
		m_volumetrics_program->set_uniform("u_Dither", (int32_t)m_dither);

		render_fullscreen_triangle(renderer, view, m_volumetrics_program.get(), tex_unit, m_flags);

		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);  
		glDisable(GL_BLEND);

		glPopDebugGroup();
	}
}

// -----------------------------------------------------------------------------------------------------------------------------------

void VolumetricLightNode::blur(Renderer* renderer, Scene* scene, View* view)
{
	glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "Blur");

	glPopDebugGroup();
}

// -----------------------------------------------------------------------------------------------------------------------------------

void VolumetricLightNode::upscale(Renderer* renderer, Scene* scene, View* view)
{
	glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "Upscale");

	glPopDebugGroup();
}

// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble