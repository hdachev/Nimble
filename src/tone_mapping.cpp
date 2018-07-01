#include "tone_mapping.h"
#include "global_graphics_resources.h"
#include "constants.h"
#include "logger.h"

// -----------------------------------------------------------------------------------------------------------------------------------

ToneMapping::ToneMapping()
{
	m_current_operator = TONE_MAPPING_REINHARD;
	m_exposure = 1.0f;
	m_uc2_exposure_bias = 2.0f;
}

// -----------------------------------------------------------------------------------------------------------------------------------

ToneMapping::~ToneMapping() {}

// -----------------------------------------------------------------------------------------------------------------------------------

void ToneMapping::initialize(uint16_t width, uint16_t height)
{
	on_window_resized(width, height);

	std::string vs_path = "shader/post_process/quad_vs.glsl";
	m_tone_mapping_vs = GlobalGraphicsResources::load_shader(GL_VERTEX_SHADER, vs_path);

	std::string fs_path = "shader/post_process/tone_mapping/tone_mapping_fs.glsl";
	m_tone_mapping_fs = GlobalGraphicsResources::load_shader(GL_FRAGMENT_SHADER, fs_path);

	dw::Shader* shaders[] = { m_tone_mapping_vs, m_tone_mapping_fs };
	std::string combined_path = vs_path + fs_path;
	m_tone_mapping_program = GlobalGraphicsResources::load_program(combined_path, 2, &shaders[0]);

	if (!m_tone_mapping_vs || !m_tone_mapping_fs || !m_tone_mapping_program)
	{
		DW_LOG_ERROR("Failed to load Tone Mapping shaders");
	}
}

// -----------------------------------------------------------------------------------------------------------------------------------

void ToneMapping::shutdown() {}

// -----------------------------------------------------------------------------------------------------------------------------------

void ToneMapping::on_window_resized(uint16_t width, uint16_t height)
{
	// Clear earlier render targets.
	GlobalGraphicsResources::destroy_framebuffer(FRAMEBUFFER_TONE_MAPPING);
	GlobalGraphicsResources::destroy_texture(RENDER_TARGET_TONE_MAPPING);

	// Create Render targets.
	m_tone_mapped_rt = GlobalGraphicsResources::create_texture_2d(RENDER_TARGET_TONE_MAPPING, width, height, GL_RGB8, GL_RGB, GL_UNSIGNED_BYTE);
	m_tone_mapped_rt->set_min_filter(GL_LINEAR);

	// Create FBO.
	m_tone_mapped_fbo = GlobalGraphicsResources::create_framebuffer(FRAMEBUFFER_TONE_MAPPING);

	// Attach render target to FBO.
	m_tone_mapped_fbo->attach_render_target(0, m_tone_mapped_rt, 0, 0);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void ToneMapping::render(uint32_t w, uint32_t h)
{
	m_tone_mapping_program->use();

	m_tone_mapping_program->set_uniform("s_CurrentOperator", m_current_operator);
	m_tone_mapping_program->set_uniform("s_Exposure", m_exposure);
	m_tone_mapping_program->set_uniform("s_ExposureBias", m_uc2_exposure_bias);

	if (m_tone_mapping_program->set_uniform("s_Color", 0))
		GlobalGraphicsResources::lookup_texture(RENDER_TARGET_MOTION_BLUR)->bind(0);

	m_post_process_renderer.render(w, h, m_tone_mapped_fbo);
}

// -----------------------------------------------------------------------------------------------------------------------------------