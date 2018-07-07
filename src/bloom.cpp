#include "bloom.h"
#include "global_graphics_resources.h"
#include "constants.h"
#include "logger.h"
#include "gpu_profiler.h"

#include <imgui.h>

#define RT_SCALE_2 0
#define RT_SCALE_4 1
#define RT_SCALE_8 2
#define RT_SCALE_16 3

// -----------------------------------------------------------------------------------------------------------------------------------

Bloom::Bloom()
{
	m_enabled = true;
	m_threshold = 1.0f;
	m_strength = 1.0f;
}

// -----------------------------------------------------------------------------------------------------------------------------------

Bloom::~Bloom()
{

}

// -----------------------------------------------------------------------------------------------------------------------------------

void Bloom::initialize(uint16_t width, uint16_t height)
{
	on_window_resized(width, height);

	std::string vs_path = "shader/post_process/quad_vs.glsl";
	m_quad_vs = GlobalGraphicsResources::load_shader(GL_VERTEX_SHADER, vs_path);

	{
		std::string fs_path = "shader/post_process/bloom/bright_pass_fs.glsl";
		m_bright_pass_fs = GlobalGraphicsResources::load_shader(GL_FRAGMENT_SHADER, fs_path);

		dw::Shader* shaders[] = { m_quad_vs, m_bright_pass_fs };
		std::string combined_path = vs_path + fs_path;
		m_bright_pass_program = GlobalGraphicsResources::load_program(combined_path, 2, &shaders[0]);

		if (!m_quad_vs || !m_bright_pass_fs || !m_bright_pass_program)
		{
			DW_LOG_ERROR("Failed to load Bloom bright pass shaders");
		}
	}

	{
		std::string fs_path = "shader/post_process/downsample_fs.glsl";
		m_bloom_downsample_fs = GlobalGraphicsResources::load_shader(GL_FRAGMENT_SHADER, fs_path);

		dw::Shader* shaders[] = { m_quad_vs, m_bloom_downsample_fs };
		std::string combined_path = vs_path + fs_path;
		m_bloom_downsample_program = GlobalGraphicsResources::load_program(combined_path, 2, &shaders[0]);

		if (!m_quad_vs || !m_bloom_downsample_fs || !m_bloom_downsample_program)
		{
			DW_LOG_ERROR("Failed to load Bloom downsample pass shaders");
		}
	}

	{
		std::string fs_path = "shader/post_process/gaussian_blur_fs.glsl";
		m_bloom_blur_fs = GlobalGraphicsResources::load_shader(GL_FRAGMENT_SHADER, fs_path);

		dw::Shader* shaders[] = { m_quad_vs, m_bloom_blur_fs };
		std::string combined_path = vs_path + fs_path;
		m_bloom_blur_program = GlobalGraphicsResources::load_program(combined_path, 2, &shaders[0]);

		if (!m_quad_vs || !m_bloom_blur_fs || !m_bloom_blur_program)
		{
			DW_LOG_ERROR("Failed to load Bloom blur pass shaders");
		}
	}

	{
		std::string fs_path = "shader/post_process/bloom/composite_fs.glsl";
		m_bloom_composite_fs = GlobalGraphicsResources::load_shader(GL_FRAGMENT_SHADER, fs_path);

		dw::Shader* shaders[] = { m_quad_vs, m_bloom_composite_fs };
		std::string combined_path = vs_path + fs_path;
		m_bloom_composite_program = GlobalGraphicsResources::load_program(combined_path, 2, &shaders[0]);

		if (!m_quad_vs || !m_bloom_composite_fs || !m_bloom_composite_program)
		{
			DW_LOG_ERROR("Failed to load Bloom blur pass shaders");
		}
	}
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Bloom::shutdown()
{

}

// -----------------------------------------------------------------------------------------------------------------------------------

void Bloom::profiling_gui()
{
	ImGui::Text("Bloom - Bright Pass: %f ms", GPUProfiler::result("Bloom - Bright Pass"));
	ImGui::Text("Bloom - Downsample: %f ms", GPUProfiler::result("Bloom - Downsample"));
	ImGui::Text("Bloom - Blur: %f ms", GPUProfiler::result("Bloom - Blur"));
	ImGui::Text("Bloom - Composite: %f ms", GPUProfiler::result("Bloom - Composite"));
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Bloom::on_window_resized(uint16_t width, uint16_t height)
{
	GlobalGraphicsResources::destroy_framebuffer(FRAMEBUFFER_BRIGHT_PASS);
	GlobalGraphicsResources::destroy_texture(RENDER_TARGET_BRIGHT_PASS);

	m_bright_pass_rt = GlobalGraphicsResources::create_texture_2d(RENDER_TARGET_BRIGHT_PASS, width, height, GL_RGB16F, GL_RGB, GL_HALF_FLOAT);
	m_bright_pass_rt->set_min_filter(GL_LINEAR);
	m_bright_pass_rt->set_wrapping(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);

	m_bright_pass_fbo = GlobalGraphicsResources::create_framebuffer(FRAMEBUFFER_BRIGHT_PASS);
	m_bright_pass_fbo->attach_render_target(0, m_bright_pass_rt, 0, 0);

	GlobalGraphicsResources::destroy_framebuffer(FRAMEBUFFER_BLOOM_COMPOSITE);
	GlobalGraphicsResources::destroy_texture(RENDER_TARGET_BLOOM_COMPOSITE);

	m_composite_rt = GlobalGraphicsResources::create_texture_2d(RENDER_TARGET_BLOOM_COMPOSITE, width, height, GL_RGB16F, GL_RGB, GL_HALF_FLOAT);
	m_composite_rt->set_min_filter(GL_LINEAR);
	m_composite_rt->set_wrapping(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);

	m_composite_fbo = GlobalGraphicsResources::create_framebuffer(FRAMEBUFFER_BLOOM_COMPOSITE);
	m_composite_fbo->attach_render_target(0, m_composite_rt, 0, 0);

	// Clear earlier render targets.
	for (uint32_t i = 0; i < 4; i++)
	{
		for (uint32_t j = 0; j < 2; j++)
		{
			uint32_t scale = pow(2, i + 1);

			std::string fbo_name = "BLOOM";
			fbo_name += std::to_string(scale);
			fbo_name += "_FBO_";
			fbo_name += std::to_string(j + 1);

			std::string rt_name = "BLOOM";
			rt_name += std::to_string(scale);
			rt_name += "_RT_";
			rt_name += std::to_string(j + 1);

			GlobalGraphicsResources::destroy_framebuffer(fbo_name);
			GlobalGraphicsResources::destroy_texture(rt_name);

			// Create render target.
			m_bloom_rt[i][j] = GlobalGraphicsResources::create_texture_2d(rt_name, width / scale, height / scale, GL_RGB16F, GL_RGB, GL_HALF_FLOAT);
			m_bloom_rt[i][j]->set_min_filter(GL_LINEAR);
			m_bloom_rt[i][j]->set_wrapping(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);

			// Create FBO.
			m_bloom_fbo[i][j] = GlobalGraphicsResources::create_framebuffer(fbo_name);

			// Attach render target to FBO.
			m_bloom_fbo[i][j]->attach_render_target(0, m_bloom_rt[i][j], 0, 0);
		}
	}
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Bloom::render(uint32_t w, uint32_t h)
{
	if (m_enabled || m_strength > 0)
	{
		bright_pass(w, h);
		downsample(w, h);
		blur(w, h);
	}
	
	composite(w, h);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Bloom::bright_pass(uint32_t w, uint32_t h)
{
	GPUProfiler::begin("Bloom - Bright Pass");

	m_bright_pass_program->use();

	if (m_bright_pass_program->set_uniform("s_Color", 0))
		GlobalGraphicsResources::lookup_texture(RENDER_TARGET_MOTION_BLUR)->bind(0);

	m_bright_pass_program->set_uniform("u_Threshold", m_threshold);

	m_post_process_renderer.render(w, h, m_bright_pass_fbo);

	GPUProfiler::end("Bloom - Bright Pass");
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Bloom::downsample(uint32_t w, uint32_t h)
{
	GPUProfiler::begin("Bloom - Downsample");

	m_bloom_downsample_program->use();

	// Progressively blur bright pass into blur textures.
	for (uint32_t i = 0; i < 4; i++)
	{
		uint32_t scale = pow(2, i + 1);

		if (m_bloom_downsample_program->set_uniform("s_Texture", 0))
		{
			// If this is the initial downsample, use the bright pass texture.
			if (i == 0)
				m_bright_pass_rt->bind(0);
			else // Else you the output of the previous pass
				m_bloom_rt[i - 1][0]->bind(0);
		}

		m_post_process_renderer.render(w / scale, h / scale, m_bloom_fbo[i][0]);
	}

	GPUProfiler::end("Bloom - Downsample");
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Bloom::blur(uint32_t w, uint32_t h)
{
	GPUProfiler::begin("Bloom - Blur");

	// Bloom each downsampled target
	for (uint32_t i = 0; i < 4; i++)
	{
		uint32_t scale = pow(2, i + 1);

		// Each downsampled target is horizontally blurred into the second Render target from the same scale and 
		// vertically blurred back into the original Render target.
		separable_blur(m_bloom_rt[i][0], m_bloom_fbo[i][0], m_bloom_rt[i][1], m_bloom_fbo[i][1], w / scale, h / scale);
	}

	GPUProfiler::end("Bloom - Blur");
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Bloom::composite(uint32_t w, uint32_t h)
{
	GPUProfiler::begin("Bloom - Composite");

	m_bloom_composite_program->use();

	m_bloom_composite_program->set_uniform("u_Strength", m_enabled ? m_strength : 0.0f);

	if (m_bloom_composite_program->set_uniform("s_Color", 0))
		GlobalGraphicsResources::lookup_texture(RENDER_TARGET_MOTION_BLUR)->bind(0);

	if (m_bloom_composite_program->set_uniform("s_Bloom2", 1))
		m_bloom_rt[RT_SCALE_2][0]->bind(1);

	if (m_bloom_composite_program->set_uniform("s_Bloom4", 2))
		m_bloom_rt[RT_SCALE_4][0]->bind(2);

	if (m_bloom_composite_program->set_uniform("s_Bloom8", 3))
		m_bloom_rt[RT_SCALE_8][0]->bind(3);

	if (m_bloom_composite_program->set_uniform("s_Bloom16", 4))
		m_bloom_rt[RT_SCALE_16][0]->bind(4);

	m_post_process_renderer.render(w, h, m_composite_fbo);

	GPUProfiler::end("Bloom - Composite");
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Bloom::separable_blur(dw::Texture* src_rt, dw::Framebuffer* dest_fbo, dw::Texture* temp_rt, dw::Framebuffer* temp_fbo, uint32_t w, uint32_t h)
{
	m_bloom_blur_program->use();

	// Horizontal blur
	if (m_bloom_blur_program->set_uniform("s_Texture", 0))
		src_rt->bind(0);

    glm::vec2 direction = glm::vec2(1, 0);
    m_bloom_blur_program->set_uniform("u_Direction", direction);
    
    glm::vec2 resolution = glm::vec2(w, h);
    m_bloom_blur_program->set_uniform("u_Resolution", resolution);
    
	m_post_process_renderer.render(w, h, temp_fbo);

	// Vertical blur
	temp_rt->bind(0);
    
    direction = glm::vec2(0, 1);
    m_bloom_blur_program->set_uniform("u_Direction", direction);

	m_post_process_renderer.render(w, h, dest_fbo);
}

// -----------------------------------------------------------------------------------------------------------------------------------
