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

    m_volumetrics_rt = register_scaled_output_render_target("Volumetrics", 1.0f, 1.0f, GL_TEXTURE_2D, GL_RGB16F, GL_RGB, GL_HALF_FLOAT, 1, 1);
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool VolumetricLightNode::initialize(Renderer* renderer, ResourceManager* res_mgr)
{
    m_color_rt = find_input_render_target("Color");
	m_depth_rt = find_input_render_target("Depth");

	m_volumetrics_rtv = RenderTargetView(0, 0, 0, m_volumetrics_rt->texture);

    m_volumetrics_vs = res_mgr->load_shader("shader/post_process/fullscreen_triangle_vs.glsl", GL_VERTEX_SHADER);
    m_volumetrics_fs = res_mgr->load_shader("shader/post_process/volumetric_light/volumetric_light_fs.glsl", GL_FRAGMENT_SHADER);

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

    m_volumetrics_program->use();

    renderer->bind_render_targets(1, &m_volumetrics_rtv, nullptr);
    glViewport(0, 0, m_graph->window_width(), m_graph->window_height());

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

	int32_t tex_unit = 0;

    if (m_volumetrics_program->set_uniform("s_Depth", tex_unit))
        m_depth_rt->texture->bind(tex_unit++);
	
	float g_2 = m_mie_g * m_mie_g;
	m_volumetrics_program->set_uniform("u_MieG", glm::vec4(1.0f - g_2, 1.0f + g_2, 2.0f * m_mie_g, 1.0f / (4.0f * M_PI)));
	m_volumetrics_program->set_uniform("u_NumSamples", m_num_samples);

    render_fullscreen_triangle(renderer, view, m_volumetrics_program.get(), tex_unit, NODE_USAGE_SHADOW_MAP);
}

// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble