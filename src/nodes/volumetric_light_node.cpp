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
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool VolumetricLightNode::initialize(Renderer* renderer, ResourceManager* res_mgr)
{
    m_depth_rt = find_input_render_target("Depth");

    m_volumetrics_rtv = RenderTargetView(0, 0, 0, m_color_rt->texture);

    m_volumetrics_vs = res_mgr->load_shader("shader/post_process/fullscreen_triangle_vs.glsl", GL_VERTEX_SHADER);
    m_volumetrics_fs = res_mgr->load_shader("shader/post_process/volumetric_light/volumetric_light_fs.glsl", GL_FRAGMENT_SHADER, m_flags, renderer);

    if (m_volumetrics_vs && m_volumetrics_fs)
    {
        m_volumetrics_program = renderer->create_program(m_volumetrics_vs, m_volumetrics_fs);
        return true;
    }
    else
        return false;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void VolumetricLightNode::execute(double delta, Renderer* renderer, Scene* scene, View* view)
{
    volumetrics(renderer, scene, view);
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

    float g_2 = m_mie_g * m_mie_g;
    m_volumetrics_program->set_uniform("u_MieG", glm::vec4(1.0f - g_2, 1.0f + g_2, 2.0f * m_mie_g, 1.0f / (4.0f * M_PI)));
    m_volumetrics_program->set_uniform("u_NumSamples", m_num_samples);

	render_fullscreen_triangle(renderer, view, m_volumetrics_program.get(), tex_unit, m_flags);

	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);  
	glDisable(GL_BLEND);
}

// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble