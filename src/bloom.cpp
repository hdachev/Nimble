#include "bloom.h"
#include "global_graphics_resources.h"
#include "constants.h"
#include "logger.h"
#include "gpu_profiler.h"

#include <imgui.h>

// -----------------------------------------------------------------------------------------------------------------------------------

Bloom::Bloom()
{
	m_enabled = true;
	m_threshold = 1.0f;
	m_strength = 0.65f;
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
		std::string fs_path = "shader/post_process/bloom/downsample_fs.glsl";
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
		std::string fs_path = "shader/post_process/bloom/upsample_fs.glsl";
		m_bloom_upsample_fs = GlobalGraphicsResources::load_shader(GL_FRAGMENT_SHADER, fs_path);

		dw::Shader* shaders[] = { m_quad_vs, m_bloom_upsample_fs };
		std::string combined_path = vs_path + fs_path;
		m_bloom_upsample_program = GlobalGraphicsResources::load_program(combined_path, 2, &shaders[0]);

		if (!m_quad_vs || !m_bloom_upsample_fs || !m_bloom_upsample_program)
		{
			DW_LOG_ERROR("Failed to load Bloom Upsample pass shaders");
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
	ImGui::Text("Bloom - Upsample: %f ms", GPUProfiler::result("Bloom - Upsample"));
	ImGui::Text("Bloom - Composite: %f ms", GPUProfiler::result("Bloom - Composite"));
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Bloom::on_window_resized(uint16_t width, uint16_t height)
{
	GlobalGraphicsResources::destroy_framebuffer(FRAMEBUFFER_BLOOM_COMPOSITE);
	GlobalGraphicsResources::destroy_texture(RENDER_TARGET_BLOOM_COMPOSITE);

	m_composite_rt = GlobalGraphicsResources::create_texture_2d(RENDER_TARGET_BLOOM_COMPOSITE, width, height, GL_RGB32F, GL_RGB, GL_FLOAT);
	m_composite_rt->set_min_filter(GL_LINEAR);
	m_composite_rt->set_wrapping(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);

	m_composite_fbo = GlobalGraphicsResources::create_framebuffer(FRAMEBUFFER_BLOOM_COMPOSITE);
	m_composite_fbo->attach_render_target(0, m_composite_rt, 0, 0);

	// Clear earlier render targets.
	for (uint32_t i = 0; i < BLOOM_TEX_CHAIN_SIZE; i++)
	{
		uint32_t scale = pow(2, i);

		std::string fbo_name = "BLOOM";
		fbo_name += std::to_string(scale);
		fbo_name += "_FBO_";
		fbo_name += std::to_string(i + 1);

		std::string rt_name = "BLOOM";
		rt_name += std::to_string(scale);
		rt_name += "_RT_";
		rt_name += std::to_string(i + 1);

		GlobalGraphicsResources::destroy_framebuffer(fbo_name);
		GlobalGraphicsResources::destroy_texture(rt_name);

		// Create render target.
		m_bloom_rt[i] = GlobalGraphicsResources::create_texture_2d(rt_name, width / scale, height / scale, GL_RGB32F, GL_RGB, GL_FLOAT);
		m_bloom_rt[i]->set_min_filter(GL_LINEAR);
		m_bloom_rt[i]->set_wrapping(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);

		// Create FBO.
		m_bloom_fbo[i] = GlobalGraphicsResources::create_framebuffer(fbo_name);

		// Attach render target to FBO.
		m_bloom_fbo[i]->attach_render_target(0, m_bloom_rt[i], 0, 0);
	}
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Bloom::render(uint32_t w, uint32_t h)
{
	if (m_enabled || m_strength > 0)
	{
		bright_pass(w, h);
		downsample(w, h);
		upsample(w, h);
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

	m_post_process_renderer.render(w, h, m_bloom_fbo[0]);

	GPUProfiler::end("Bloom - Bright Pass");
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Bloom::downsample(uint32_t w, uint32_t h)
{
	GPUProfiler::begin("Bloom - Downsample");

	m_bloom_downsample_program->use();

	// Progressively blur bright pass into blur textures.
	for (uint32_t i = 0; i < (BLOOM_TEX_CHAIN_SIZE - 1); i++)
	{
		float scale = pow(2, i + 1);

		glm::vec2 pixel_size = glm::vec2(1.0f / (float(w) / scale), 1.0f / (float(h) / scale));
		m_bloom_downsample_program->set_uniform("u_PixelSize", pixel_size);

		if (m_bloom_downsample_program->set_uniform("s_Texture", 0))
			m_bloom_rt[i]->bind(0);

		m_post_process_renderer.render(w / scale, h / scale, m_bloom_fbo[i + 1]);
	}

	GPUProfiler::end("Bloom - Downsample");
}

// -----------------------------------------------------------------------------------------------------------------------------------

// TODO: Prevent clearing when upsampling and use additive blending.
void Bloom::upsample(uint32_t w, uint32_t h)
{
	GPUProfiler::begin("Bloom - Upsample");

	m_bloom_upsample_program->use();

	m_bloom_upsample_program->set_uniform("u_Strength", m_enabled ? m_strength : 0.0f);

	//glEnable(GL_BLEND);

	//glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
	//glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ZERO);

	// Upsample each downsampled target
	for (uint32_t i = 0; i < (BLOOM_TEX_CHAIN_SIZE - 1); i++)
	{
		float scale = pow(2, BLOOM_TEX_CHAIN_SIZE - i - 2);

		glm::vec2 pixel_size = glm::vec2(1.0f / (float(w) / scale), 1.0f / (float(h) / scale));
		m_bloom_upsample_program->set_uniform("u_PixelSize", pixel_size);
		
		if (m_bloom_upsample_program->set_uniform("s_Texture", 0))
			m_bloom_rt[BLOOM_TEX_CHAIN_SIZE - i - 1]->bind(0);

		m_post_process_renderer.render(w / scale, h / scale, m_bloom_fbo[BLOOM_TEX_CHAIN_SIZE - i - 2]);
	}

	//glDisable(GL_BLEND);

	GPUProfiler::end("Bloom - Upsample");
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Bloom::composite(uint32_t w, uint32_t h)
{
	GPUProfiler::begin("Bloom - Composite");

	m_bloom_composite_program->use();

	m_bloom_composite_program->set_uniform("u_Strength", m_enabled ? m_strength : 0.0f);

	if (m_bloom_composite_program->set_uniform("s_Color", 0))
		GlobalGraphicsResources::lookup_texture(RENDER_TARGET_MOTION_BLUR)->bind(0);

	if (m_bloom_composite_program->set_uniform("s_Bloom", 1))
		m_bloom_rt[0]->bind(1);
	
	m_post_process_renderer.render(w, h, m_composite_fbo);

	GPUProfiler::end("Bloom - Composite");
}

// -----------------------------------------------------------------------------------------------------------------------------------
