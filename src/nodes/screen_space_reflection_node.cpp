#include "screen_space_reflection_node.h"
#include "../render_graph.h"
#include "../resource_manager.h"
#include "../renderer.h"
#include "../profiler.h"

#define NUM_SSR_THREADS 8

namespace nimble
{
DEFINE_RENDER_NODE_FACTORY(ScreenSpaceReflectionNode)

// -----------------------------------------------------------------------------------------------------------------------------------

ScreenSpaceReflectionNode::ScreenSpaceReflectionNode(RenderGraph* graph) :
    RenderNode(graph)
{
    m_enabled   = true;
}

// -----------------------------------------------------------------------------------------------------------------------------------

ScreenSpaceReflectionNode::~ScreenSpaceReflectionNode()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void ScreenSpaceReflectionNode::declare_connections()
{
    register_input_render_target("HiZDepth");
    register_input_render_target("Metallic");
    register_input_render_target("Normal");

    m_ssr_rt = register_scaled_output_render_target("SSR", 1.0f, 1.0f, GL_TEXTURE_2D, GL_RGB16F, GL_RGB, GL_HALF_FLOAT);
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool ScreenSpaceReflectionNode::initialize(Renderer* renderer, ResourceManager* res_mgr)
{
    register_bool_parameter("Enabled", m_enabled);

	m_hiz_depth_rt = find_input_render_target("HiZDepth");
    m_metallic_rt  = find_input_render_target("Metallic");
    m_normal_rt    = find_input_render_target("Normal");

    m_ssr_rtv = RenderTargetView(0, 0, 0, m_ssr_rt->texture);

    m_ssr_cs = res_mgr->load_shader("shader/post_process/screen_space_reflections/ssr_cs.glsl", GL_COMPUTE_SHADER);

    if (m_ssr_cs)
    {
        m_ssr_program = renderer->create_program({ m_ssr_cs });
        return true;
    }
    else
        return false;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void ScreenSpaceReflectionNode::execute(double delta, Renderer* renderer, Scene* scene, View* view)
{
    if (m_enabled)
    {
        m_ssr_program->use();

		m_ssr_program->set_uniform("u_HiZLevels", (int32_t)m_hiz_depth_rt->texture->mip_levels());

		if (m_ssr_program->set_uniform("s_HiZDepth", 0))
            m_hiz_depth_rt->texture->bind(0);

		if (m_ssr_program->set_uniform("s_Metallic", 1))
			m_metallic_rt->texture->bind(1);

		if (m_ssr_program->set_uniform("s_Normal", 2))
			m_normal_rt->texture->bind(2);

		m_ssr_rt->texture->bind_image(0, 0, 0, GL_WRITE_ONLY, GL_RGB16F);

		glDispatchCompute(m_graph->window_width() / NUM_SSR_THREADS, m_graph->window_height() / NUM_SSR_THREADS, 1);
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void ScreenSpaceReflectionNode::shutdown()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::string ScreenSpaceReflectionNode::name()
{
    return "Screen Space Reflections";
}

// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble