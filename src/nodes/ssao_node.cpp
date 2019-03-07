#include "ssao_node.h"
#include "../render_graph.h"
#include "../resource_manager.h"
#include "../renderer.h"
#include "../logger.h"
#include <random>
#define GLM_ENABLE_EXPERIMENTAL
#include <gtx/compatibility.hpp>

namespace nimble
{
DEFINE_RENDER_NODE_FACTORY(SSAONode)

struct SSAOData
{
	glm::vec4 kernel[64];
};

// -----------------------------------------------------------------------------------------------------------------------------------

SSAONode::SSAONode(RenderGraph* graph) :
    RenderNode(graph)
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

SSAONode::~SSAONode()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void SSAONode::declare_connections()
{
	register_input_render_target("Normals");
    register_input_render_target("Depth");

    m_ssao_intermediate_rt = register_scaled_intermediate_render_target("SSAO_Intermediate", 0.5f, 0.5f, GL_TEXTURE_2D, GL_R32F, GL_RED, GL_FLOAT);
    m_ssao_rt = register_scaled_output_render_target("SSAO", 0.5f, 0.5f, GL_TEXTURE_2D, GL_R32F, GL_RED, GL_FLOAT);
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool SSAONode::initialize(Renderer* renderer, ResourceManager* res_mgr)
{
	m_ssao_intermediate_rtv = RenderTargetView(0, 0, 0, m_ssao_intermediate_rt->texture);
	m_ssao_rtv = RenderTargetView(0, 0, 0, m_ssao_rt->texture);

	std::uniform_real_distribution<float> random_floats(0.0, 1.0);
	std::default_random_engine generator;
	
	std::vector<glm::vec3> ssao_noise;
	
	for (uint32_t i = 0; i < 16; i++)
	{
	    glm::vec3 noise(random_floats(generator) * 2.0f - 1.0f, random_floats(generator) * 2.0f - 1.0f, 0.0f);
	    ssao_noise.push_back(noise);
	}
	
	m_noise_texture = std::make_unique<Texture2D>(4, 4, 1, 1, 1, GL_RGB16F, GL_RGB, GL_FLOAT);
	m_noise_texture->set_min_filter(GL_NEAREST);
	m_noise_texture->set_mag_filter(GL_NEAREST);
	m_noise_texture->set_wrapping(GL_REPEAT, GL_REPEAT, GL_REPEAT);
	m_noise_texture->set_data(0, 0, ssao_noise.data());
	
	std::vector<glm::vec4> ssao_kernel;
	
	for (uint32_t i = 0; i < 64; i++)
	{
	    glm::vec4 sample = glm::vec4(random_floats(generator) * 2.0f - 1.0f, random_floats(generator) * 2.0f - 1.0f, random_floats(generator), 0.0f);
	    sample = glm::normalize(sample);
	    sample *= random_floats(generator);
	
	    float scale = float(i) / 64.0f;
	    scale = glm::lerp(0.1f, 1.0f, scale * scale);
	    sample *= scale;
	
	    ssao_kernel.push_back(sample);
	}
	
	m_kernel_ubo = std::make_unique<UniformBuffer>(GL_DYNAMIC_DRAW, sizeof(SSAOData), ssao_kernel.data());
	
	m_normals_rt = find_input_render_target("Normals");
	m_depth_rt = find_input_render_target("Depth");
	
	m_triangle_vs = res_mgr->load_shader("shader/post_process/fullscreen_triangle_vs.glsl", GL_VERTEX_SHADER);
	m_ssao_fs = res_mgr->load_shader("shader/post_process/ssao/ssao_fs.glsl", GL_FRAGMENT_SHADER);
	m_ssao_blur_fs  = res_mgr->load_shader("shader/post_process/ssao/ssao_blur_fs.glsl", GL_FRAGMENT_SHADER);
	
	if (m_triangle_vs)
	{
		if (m_ssao_fs)
		{
			m_ssao_program = renderer->create_program(m_triangle_vs, m_ssao_fs);
			m_ssao_program->uniform_block_binding("u_SSAOData", 0);
		}
	    else
	        return false;
	
	    if (m_ssao_blur_fs)
	        m_ssao_blur_program = renderer->create_program(m_triangle_vs, m_ssao_blur_fs);
	    else
	        return false;
	
	    return true;
	}
	else
	    return false;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void SSAONode::execute(Renderer* renderer, Scene* scene, View* view)
{
	ssao(renderer, view);
	blur(renderer);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void SSAONode::shutdown()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::string SSAONode::name()
{
    return "SSAO";
}

// -----------------------------------------------------------------------------------------------------------------------------------

void SSAONode::ssao(Renderer* renderer, View* view)
{
	m_ssao_program->use();

   m_kernel_ubo->bind_base(0);

   if (m_ssao_program->set_uniform("s_Normals", 0))
	   m_normals_rt->texture->bind(0);

	if (m_ssao_program->set_uniform("s_Depth", 1))
	    m_depth_rt->texture->bind(1);

   if (m_ssao_program->set_uniform("s_Noise", 2))
	    m_noise_texture->bind(2);

   m_ssao_program->set_uniform("u_ViewportSize", glm::vec2(m_graph->window_width(), m_graph->window_height()));
   m_ssao_program->set_uniform("u_NumSamples", m_num_samples);
   m_ssao_program->set_uniform("u_Radius", m_radius);
   m_ssao_program->set_uniform("u_Bias", m_bias);

   renderer->bind_render_targets(1, &m_ssao_intermediate_rtv, nullptr);
   glViewport(0, 0, m_graph->window_width()/2, m_graph->window_height()/2);
   glClear(GL_COLOR_BUFFER_BIT);

   render_fullscreen_triangle(renderer, view, nullptr, 0, NODE_USAGE_PER_VIEW_UBO);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void SSAONode::blur(Renderer* renderer)
{
	m_ssao_blur_program->use();

    if (m_ssao_blur_program->set_uniform("s_SSAO", 0))
	    m_ssao_intermediate_rt->texture->bind(0);

    renderer->bind_render_targets(1, &m_ssao_rtv, nullptr);
    glViewport(0, 0, m_graph->window_width()/2, m_graph->window_height()/2);
    glClear(GL_COLOR_BUFFER_BIT);

    render_fullscreen_triangle(renderer, nullptr);
}

// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble