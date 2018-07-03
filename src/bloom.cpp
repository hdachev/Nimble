#include "bloom.h"
#include "global_graphics_resources.h"
#include "constants.h"
#include "logger.h"

#define RT_SCALE_2 0
#define RT_SCALE_4 1
#define RT_SCALE_8 2
#define RT_SCALE_16 3

// -----------------------------------------------------------------------------------------------------------------------------------

Bloom::Bloom()
{

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
	m_bloom_downsample_vs = GlobalGraphicsResources::load_shader(GL_VERTEX_SHADER, vs_path);
	m_bloom_blur_vs = GlobalGraphicsResources::load_shader(GL_VERTEX_SHADER, vs_path);
	m_bloom_composite_vs = GlobalGraphicsResources::load_shader(GL_VERTEX_SHADER, vs_path);

	{
		std::string fs_path = "shader/post_process/bloom/downsample_fs.glsl";
		m_bloom_downsample_fs = GlobalGraphicsResources::load_shader(GL_FRAGMENT_SHADER, fs_path);

		dw::Shader* shaders[] = { m_bloom_downsample_vs, m_bloom_downsample_fs };
		std::string combined_path = vs_path + fs_path;
		m_bloom_downsample_program = GlobalGraphicsResources::load_program(combined_path, 2, &shaders[0]);

		if (!m_bloom_downsample_vs || !m_bloom_downsample_fs || !m_bloom_downsample_program)
		{
			DW_LOG_ERROR("Failed to load Bloom downsample pass shaders");
		}
	}

	{
		std::string fs_path = "shader/post_process/bloom/blur_fs.glsl";
		m_bloom_blur_fs = GlobalGraphicsResources::load_shader(GL_FRAGMENT_SHADER, fs_path);

		dw::Shader* shaders[] = { m_bloom_blur_vs, m_bloom_blur_fs };
		std::string combined_path = vs_path + fs_path;
		m_bloom_blur_program = GlobalGraphicsResources::load_program(combined_path, 2, &shaders[0]);

		if (!m_bloom_blur_vs || !m_bloom_blur_fs || !m_bloom_blur_program)
		{
			DW_LOG_ERROR("Failed to load Bloom blur pass shaders");
		}
	}

	{
		std::string fs_path = "shader/post_process/bloom/composite_fs.glsl";
		m_bloom_composite_fs = GlobalGraphicsResources::load_shader(GL_FRAGMENT_SHADER, fs_path);

		dw::Shader* shaders[] = { m_bloom_composite_vs, m_bloom_composite_fs };
		std::string combined_path = vs_path + fs_path;
		m_bloom_composite_program = GlobalGraphicsResources::load_program(combined_path, 2, &shaders[0]);

		if (!m_bloom_composite_vs || !m_bloom_composite_fs || !m_bloom_composite_program)
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

}

// -----------------------------------------------------------------------------------------------------------------------------------

void Bloom::on_window_resized(uint16_t width, uint16_t height)
{
	// Clear earlier render targets.
	for (uint32_t i = 0; i < 4; i++)
	{
		for (uint32_t j = 0; j < 2; j++)
		{
			uint32_t scale = pow(2, i + 1);

			std::string fbo_name = "FRAMEBUFFER_BLOOM";
			fbo_name += std::to_string(scale);
			fbo_name += "_";
			fbo_name += std::to_string(j + 1);

			std::string rt_name = "RENDER_TARGET_BLOOM";
			rt_name += std::to_string(2 * (i + 1));
			rt_name += "_";
			rt_name += std::to_string(j + 1);

			GlobalGraphicsResources::destroy_framebuffer(fbo_name);
			GlobalGraphicsResources::destroy_texture(rt_name);

			// Create render target.
			m_bloom_rt[i][j] = GlobalGraphicsResources::create_texture_2d(rt_name, width / scale, height / scale, GL_RGB16F, GL_RGB, GL_HALF_FLOAT);
			m_bloom_rt[i][j]->set_min_filter(GL_LINEAR);

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
	downsample(w, h);
	blur(w, h);
	composite(w, h);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Bloom::downsample(uint32_t w, uint32_t h)
{
	m_bloom_downsample_program->use();

	// Progressively blur bright pass into blur textures.
	for (uint32_t i = 0; i < 4; i++)
	{
		uint32_t scale = pow(2, i + 1);

		if (m_bloom_downsample_program->set_uniform("s_Bright", 0))
		{
			// If this is the initial downsample, use the bright pass texture.
			if (i == 0)
				GlobalGraphicsResources::lookup_texture(RENDER_TARGET_BRIGHT_PASS)->bind(0);
			else // Else you the output of the previous pass
				m_bloom_rt[i - 1][0]->bind(0);
		}

		m_post_process_renderer.render(w / scale, h / scale, m_bloom_fbo[i - 1][0]);
	}
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Bloom::blur(uint32_t w, uint32_t h)
{
	// Bloom each downsampled target
	for (uint32_t i = 0; i < 4; i++)
	{
		uint32_t scale = pow(2, i + 1);

		// Each downsampled target is horizontally blurred into the second Render target from the same scale and 
		// vertically blurred back into the original Render target.
		separable_blur(m_bloom_rt[i][0], m_bloom_fbo[i][0], m_bloom_rt[i][1], m_bloom_fbo[i][1], w / scale, h / scale);
	}
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Bloom::composite(uint32_t w, uint32_t h)
{
	m_bloom_composite_program->use();

	if (m_bloom_blur_program->set_uniform("s_BrightBlur2", 0))
		m_bloom_rt[RT_SCALE_2][0]->bind(0);

	if (m_bloom_blur_program->set_uniform("s_BrightBlur4", 0))
		m_bloom_rt[RT_SCALE_4][0]->bind(0);

	if (m_bloom_blur_program->set_uniform("s_BrightBlur8", 0))
		m_bloom_rt[RT_SCALE_8][0]->bind(0);

	if (m_bloom_blur_program->set_uniform("s_BrightBlur16", 0))
		m_bloom_rt[RT_SCALE_16][0]->bind(0);

	m_post_process_renderer.render(w / 2, h / 2, m_bloom_fbo[RT_SCALE_2][1]);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Bloom::separable_blur(dw::Texture* src_rt, dw::Framebuffer* dest_fbo, dw::Texture* temp_rt, dw::Framebuffer* temp_fbo, uint32_t w, uint32_t h)
{
	m_bloom_blur_program->use();

	// Horizontal blur
	if (m_bloom_blur_program->set_uniform("s_Bright", 0))
		src_rt->bind(0);

	m_post_process_renderer.render(w, h, temp_fbo);

	// Vertical blur
	temp_rt->bind(0);

	m_post_process_renderer.render(w, h, dest_fbo);
}

// -----------------------------------------------------------------------------------------------------------------------------------