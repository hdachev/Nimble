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
	GlobalGraphicsResources::destroy_framebuffer(FRAMEBUFFER_FORWARD);
	GlobalGraphicsResources::destroy_texture(RENDER_TARGET_FORWARD_COLOR); 
	GlobalGraphicsResources::destroy_texture(RENDER_TARGET_FORWARD_VELOCITY);
	GlobalGraphicsResources::destroy_texture(RENDER_TARGET_FORWARD_DEPTH);

	m_color_buffer = GlobalGraphicsResources::create_texture_2d(RENDER_TARGET_FORWARD_COLOR, width, height, GL_RGB16F, GL_RGB, GL_HALF_FLOAT);
	m_color_buffer->set_min_filter(GL_LINEAR);
	m_color_buffer->set_wrapping(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);

	m_velocity_buffer = GlobalGraphicsResources::create_texture_2d(RENDER_TARGET_FORWARD_VELOCITY, width, height, GL_RG16F, GL_RG, GL_HALF_FLOAT);
	m_velocity_buffer->set_min_filter(GL_LINEAR);

	m_depth_buffer = GlobalGraphicsResources::create_texture_2d(RENDER_TARGET_FORWARD_DEPTH, width, height, GL_DEPTH24_STENCIL8, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8);
	m_depth_buffer->set_min_filter(GL_LINEAR);

	m_color_fbo = GlobalGraphicsResources::create_framebuffer(FRAMEBUFFER_FORWARD);

	// Attach render targets to FBO.
	dw::Texture* render_targets[] = { m_color_buffer, m_velocity_buffer };
	m_color_fbo->attach_multiple_render_targets(2, render_targets);
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