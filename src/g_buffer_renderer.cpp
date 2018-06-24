#include "g_buffer_renderer.h"
#include "constants.h"
#include "global_graphics_resources.h"
#include "logger.h"

// -----------------------------------------------------------------------------------------------------------------------------------

GBufferRenderer::GBufferRenderer()
{

}

// -----------------------------------------------------------------------------------------------------------------------------------

GBufferRenderer::~GBufferRenderer()
{

}

// -----------------------------------------------------------------------------------------------------------------------------------

void GBufferRenderer::initialize(uint16_t width, uint16_t height)
{
	on_window_resized(width, height);

	std::string vs_path = "shader/g_buffer_vs.glsl";
	m_gbuffer_vs = GlobalGraphicsResources::load_shader(GL_VERTEX_SHADER, vs_path, nullptr);

	std::string fs_path = "shader/g_buffer_fs.glsl";
	m_gbuffer_fs = GlobalGraphicsResources::load_shader(GL_FRAGMENT_SHADER, fs_path, nullptr);

	dw::Shader* shaders[] = { m_gbuffer_vs, m_gbuffer_fs };

	m_gbuffer_program = GlobalGraphicsResources::load_program(vs_path + fs_path, 2, &shaders[0]);

	if (!m_gbuffer_vs || !m_gbuffer_fs || !m_gbuffer_program)
	{
		DW_LOG_ERROR("Failed to load G-Buffer pass shaders");
	}

	m_gbuffer_program->uniform_block_binding("u_PerFrame", 0);
	m_gbuffer_program->uniform_block_binding("u_PerEntity", 1);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void GBufferRenderer::shutdown() {}

// -----------------------------------------------------------------------------------------------------------------------------------

void GBufferRenderer::on_window_resized(uint16_t width, uint16_t height)
{
	// Clear earlier render targets.
	GlobalGraphicsResources::destroy_framebuffer(GBUFFER_FBO);
	GlobalGraphicsResources::destroy_texture(RENDER_TARGET_GBUFFER_RT0);
	GlobalGraphicsResources::destroy_texture(RENDER_TARGET_GBUFFER_RT1);
	GlobalGraphicsResources::destroy_texture(RENDER_TARGET_GBUFFER_RT2);
	GlobalGraphicsResources::destroy_texture(RENDER_TARGET_GBUFFER_RT3);
	GlobalGraphicsResources::destroy_texture(RENDER_TARGET_GBUFFER_DEPTH);

	// Create Render targets.
	m_gbuffer_rt0 = GlobalGraphicsResources::create_texture_2d(RENDER_TARGET_GBUFFER_RT0, width, height, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);
	m_gbuffer_rt0->set_min_filter(GL_LINEAR);

	m_gbuffer_rt1 = GlobalGraphicsResources::create_texture_2d(RENDER_TARGET_GBUFFER_RT1, width, height, GL_RGBA16F, GL_RGBA, GL_HALF_FLOAT);
	m_gbuffer_rt1->set_min_filter(GL_LINEAR);

	m_gbuffer_rt2 = GlobalGraphicsResources::create_texture_2d(RENDER_TARGET_GBUFFER_RT2, width, height, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);
	m_gbuffer_rt2->set_min_filter(GL_LINEAR);

	m_gbuffer_rt3 = GlobalGraphicsResources::create_texture_2d(RENDER_TARGET_GBUFFER_RT3, width, height, GL_RGB16F, GL_RGB, GL_HALF_FLOAT);
	m_gbuffer_rt3->set_min_filter(GL_LINEAR);

	m_gbuffer_depth = GlobalGraphicsResources::create_texture_2d(RENDER_TARGET_GBUFFER_DEPTH, width, height, GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT);
	m_gbuffer_depth->set_min_filter(GL_LINEAR);

	// Create FBO.
	m_gbuffer_fbo = GlobalGraphicsResources::create_framebuffer(GBUFFER_FBO);
	
	// Attach render targets to FBO.
	dw::Texture* render_targets[] = { m_gbuffer_rt0, m_gbuffer_rt1, m_gbuffer_rt2, m_gbuffer_rt3 };
	m_gbuffer_fbo->attach_multiple_render_targets(4, render_targets);
	m_gbuffer_fbo->attach_depth_stencil_target(m_gbuffer_depth, 0, 0);
}


// -----------------------------------------------------------------------------------------------------------------------------------

void GBufferRenderer::render(Scene* scene, uint32_t w, uint32_t h)
{
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);

	m_gbuffer_program->use();

	m_scene_renderer.render(scene, m_gbuffer_fbo, m_gbuffer_program, w, h);
}

// -----------------------------------------------------------------------------------------------------------------------------------