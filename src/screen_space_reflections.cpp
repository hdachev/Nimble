#include "screen_space_reflections.h"
#include "global_graphics_resources.h"
#include "constants.h"
#include "logger.h"
#include "gpu_profiler.h"
#include "imgui.h"

namespace nimble
{
	// -----------------------------------------------------------------------------------------------------------------------------------

	ScreenSpaceReflections::ScreenSpaceReflections()
	{
		m_enabled = true;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	ScreenSpaceReflections::~ScreenSpaceReflections() {}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void ScreenSpaceReflections::initialize(uint16_t width, uint16_t height)
	{
		on_window_resized(width, height);

		std::string vs_path = "shader/post_process/quad_vs.glsl";
		m_ssr_vs = GlobalGraphicsResources::load_shader(GL_VERTEX_SHADER, vs_path);

		std::string fs_path = "shader/post_process/ssr/ssr_fs.glsl";
		m_ssr_fs = GlobalGraphicsResources::load_shader(GL_FRAGMENT_SHADER, fs_path);

		Shader* shaders[] = { m_ssr_vs, m_ssr_fs };
		std::string combined_path = vs_path + fs_path;
		m_ssr_program = GlobalGraphicsResources::load_program(combined_path, 2, &shaders[0]);

		if (!m_ssr_vs || !m_ssr_fs || !m_ssr_program)
		{
			NIMBLE_LOG_ERROR("Failed to load SSR shaders");
		}
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void ScreenSpaceReflections::shutdown() {}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void ScreenSpaceReflections::profiling_gui()
	{
		ImGui::Text("SSR: %f ms", GPUProfiler::result("SSR"));
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void ScreenSpaceReflections::on_window_resized(uint16_t width, uint16_t height)
	{
		// Clear earlier render targets.
		GlobalGraphicsResources::destroy_framebuffer(FRAMEBUFFER_SSR);
		GlobalGraphicsResources::destroy_texture(RENDER_TARGET_SSR);

		// Create Render targets.
		m_ssr_rt = GlobalGraphicsResources::create_texture_2d(RENDER_TARGET_SSR, width, height, GL_RGB32F, GL_RGB, GL_HALF_FLOAT);
		m_ssr_rt->set_min_filter(GL_LINEAR);
		m_ssr_rt->set_wrapping(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);

		// Create FBO.
		m_ssr_fbo = GlobalGraphicsResources::create_framebuffer(FRAMEBUFFER_SSR);

		// Attach render target to FBO.
		m_ssr_fbo->attach_render_target(0, m_ssr_rt, 0, 0);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void ScreenSpaceReflections::render(uint32_t w, uint32_t h)
	{
		if (m_enabled)
		{
			GPUProfiler::begin("SSR");

			m_ssr_program->use();

			GlobalGraphicsResources::per_frame_ubo()->bind_base(0);

			if (m_ssr_program->set_uniform("s_Color", 0))
				GlobalGraphicsResources::lookup_texture(RENDER_TARGET_DEFERRED_COLOR)->bind(0);

			if (m_ssr_program->set_uniform("s_Normals", 1))
				GlobalGraphicsResources::lookup_texture(RENDER_TARGET_GBUFFER_RT2)->bind(1);

			if (m_ssr_program->set_uniform("s_MetalRough", 2))
				GlobalGraphicsResources::lookup_texture(RENDER_TARGET_GBUFFER_RT3)->bind(2);

			if (m_ssr_program->set_uniform("s_Depth", 3))
				GlobalGraphicsResources::lookup_texture(RENDER_TARGET_GBUFFER_DEPTH)->bind(3);

			m_post_process_renderer.render(w, h, m_ssr_fbo);

			GPUProfiler::end("SSR");
		}
		else
		{
			m_ssr_fbo->bind();
			glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
			glClear(GL_COLOR_BUFFER_BIT);
		}
	}

	// -----------------------------------------------------------------------------------------------------------------------------------
}