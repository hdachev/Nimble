#include "forward_renderer.h"
#include "scene.h"
#include "global_graphics_resources.h"
#include "constants.h"

// -----------------------------------------------------------------------------------------------------------------------------------

ForwardRenderer::ForwardRenderer() {}

// -----------------------------------------------------------------------------------------------------------------------------------

ForwardRenderer::~ForwardRenderer() {}

// -----------------------------------------------------------------------------------------------------------------------------------

void ForwardRenderer::initialize(uint16_t width, uint16_t height)
{
	on_window_resized(width, height);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void ForwardRenderer::shutdown() {}

// -----------------------------------------------------------------------------------------------------------------------------------

void ForwardRenderer::on_window_resized(uint16_t width, uint16_t height)
{
	GlobalGraphicsResources::destroy_framebuffer(FRAMEBUFFER_COLOR);
	GlobalGraphicsResources::destroy_texture(RENDER_TARGET_COLOR);
	GlobalGraphicsResources::destroy_texture(RENDER_TARGET_DEPTH);

	m_color_buffer = GlobalGraphicsResources::create_texture_2d(RENDER_TARGET_COLOR, width, height, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);
	m_color_buffer->set_min_filter(GL_LINEAR);

	m_depth_buffer = GlobalGraphicsResources::create_texture_2d(RENDER_TARGET_DEPTH, width, height, GL_DEPTH24_STENCIL8, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8);
	m_depth_buffer->set_min_filter(GL_LINEAR);

	m_color_fbo = GlobalGraphicsResources::create_framebuffer(FRAMEBUFFER_COLOR);

	m_color_fbo->attach_render_target(0, m_color_buffer, 0, 0);
	m_color_fbo->attach_depth_stencil_target(m_depth_buffer, 0, 0);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void ForwardRenderer::render(Scene* scene, uint32_t w, uint32_t h)
{
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);
	
	m_scene_renderer.render(scene, m_color_fbo, nullptr, w, h);
}

// -----------------------------------------------------------------------------------------------------------------------------------