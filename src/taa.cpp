#include "taa.h"
#include "global_graphics_resources.h"
#include "constants.h"
#include "logger.h"
#include "gpu_profiler.h"

#include <imgui.h>

// -----------------------------------------------------------------------------------------------------------------------------------

TAA::TAA()
{
	m_enabled = true;
	m_first = 1;
}

// -----------------------------------------------------------------------------------------------------------------------------------

TAA::~TAA()
{

}

// -----------------------------------------------------------------------------------------------------------------------------------

void TAA::initialize(uint16_t width, uint16_t height)
{
	on_window_resized(width, height);

	std::string vs_path = "shader/post_process/quad_vs.glsl";
	m_quad_vs = GlobalGraphicsResources::load_shader(GL_VERTEX_SHADER, vs_path);

	{
		std::string fs_path = "shader/post_process/taa/taa_fs.glsl";
		m_taa_fs = GlobalGraphicsResources::load_shader(GL_FRAGMENT_SHADER, fs_path);

		dw::Shader* shaders[] = { m_quad_vs, m_taa_fs };
		std::string combined_path = vs_path + fs_path;
		m_taa_program = GlobalGraphicsResources::load_program(combined_path, 2, &shaders[0]);

		if (!m_quad_vs || !m_taa_fs || !m_taa_program)
		{
			DW_LOG_ERROR("Failed to load TAA shaders");
		}
	}

	{
		std::string fs_path = "shader/post_process/taa/taa_hist_fs.glsl";
		m_taa_hist_fs = GlobalGraphicsResources::load_shader(GL_FRAGMENT_SHADER, fs_path);

		dw::Shader* shaders[] = { m_quad_vs, m_taa_hist_fs };
		std::string combined_path = vs_path + fs_path;
		m_taa_hist_program = GlobalGraphicsResources::load_program(combined_path, 2, &shaders[0]);

		if (!m_quad_vs || !m_taa_hist_fs || !m_taa_hist_program)
		{
			DW_LOG_ERROR("Failed to load TAA History shaders");
		}
	}

	m_taa_program->uniform_block_binding("u_PerFrame", 0);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void TAA::shutdown()
{

}

// -----------------------------------------------------------------------------------------------------------------------------------

void TAA::profiling_gui()
{
	ImGui::Text("TAA: %f ms", GPUProfiler::result("TAA"));
}

// -----------------------------------------------------------------------------------------------------------------------------------

void TAA::on_window_resized(uint16_t width, uint16_t height)
{
	GlobalGraphicsResources::destroy_framebuffer(FRAMEBUFFER_TAA);
	GlobalGraphicsResources::destroy_texture(RENDER_TARGET_TAA);

	GlobalGraphicsResources::destroy_framebuffer(FRAMEBUFFER_TAA_HIST);
	GlobalGraphicsResources::destroy_texture(RENDER_TARGET_TAA_HIST);

	m_taa_rt = GlobalGraphicsResources::create_texture_2d(RENDER_TARGET_TAA, width, height, GL_RGB32F, GL_RGB, GL_FLOAT);
	m_taa_rt->set_min_filter(GL_LINEAR);
	m_taa_rt->set_wrapping(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);

	m_taa_fbo = GlobalGraphicsResources::create_framebuffer(FRAMEBUFFER_TAA);
	m_taa_fbo->attach_render_target(0, m_taa_rt, 0, 0);

	m_taa_hist_rt = GlobalGraphicsResources::create_texture_2d(RENDER_TARGET_TAA_HIST, width, height, GL_RGB32F, GL_RGB, GL_FLOAT);
	m_taa_hist_rt->set_min_filter(GL_LINEAR);
	m_taa_hist_rt->set_wrapping(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);

	m_taa_hist_fbo = GlobalGraphicsResources::create_framebuffer(FRAMEBUFFER_TAA_HIST);
	m_taa_hist_fbo->attach_render_target(0, m_taa_hist_rt, 0, 0);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void TAA::render(uint32_t w, uint32_t h)
{
	GPUProfiler::begin("TAA");

	m_taa_program->use();

	// Bind global UBO's.
	GlobalGraphicsResources::per_frame_ubo()->bind_base(0);

	PerFrameUniforms& per_frame = GlobalGraphicsResources::per_frame_uniforms();

	if (per_frame.renderer == RENDERER_FORWARD)
	{
		if (m_taa_program->set_uniform("s_Color", 0))
			GlobalGraphicsResources::lookup_texture(RENDER_TARGET_FORWARD_COLOR)->bind(0);

		if (m_taa_program->set_uniform("s_Velocity", 2))
			GlobalGraphicsResources::lookup_texture(RENDER_TARGET_FORWARD_VELOCITY)->bind(2);
	}
	else if (per_frame.renderer == RENDERER_DEFERRED)
	{
		if (m_taa_program->set_uniform("s_Color", 0))
			GlobalGraphicsResources::lookup_texture(RENDER_TARGET_DEFERRED_COLOR)->bind(0);

		if (m_taa_program->set_uniform("s_Velocity", 2))
			GlobalGraphicsResources::lookup_texture(RENDER_TARGET_GBUFFER_RT1)->bind(2);
	}

	if (m_taa_program->set_uniform("s_History", 1))
		m_taa_hist_rt->bind(1);

	m_taa_program->set_uniform("u_Enabled", int(m_enabled));

	glm::vec2 pixel_size = glm::vec2(1.0f / float(w), 1.0f / float(h));
	m_taa_program->set_uniform("u_PixelSize", pixel_size);

	m_post_process_renderer.render(w, h, m_taa_fbo);

	// Copy TAA result into History buffer
	m_taa_hist_program->use();

	if (m_taa_hist_program->set_uniform("s_Color", 0))
		m_taa_rt->bind(0);

	m_post_process_renderer.render(w, h, m_taa_hist_fbo);

	GPUProfiler::end("TAA");
}

// -----------------------------------------------------------------------------------------------------------------------------------
