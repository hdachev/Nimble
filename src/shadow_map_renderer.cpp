#include "shadow_map_renderer.h"
#include "csm.h"
#include "global_graphics_resources.h"
#include "logger.h"

// -----------------------------------------------------------------------------------------------------------------------------------

ShadowMapRenderer::ShadowMapRenderer() 
{
	std::string vs_path = "shader/csm/csm_vs.glsl";
	m_csm_vs = GlobalGraphicsResources::load_shader(GL_VERTEX_SHADER, vs_path);

	std::string fs_path = "shader/csm/csm_fs.glsl";
	m_csm_fs = GlobalGraphicsResources::load_shader(GL_FRAGMENT_SHADER, fs_path);

	dw::Shader* shaders[] = { m_csm_vs, m_csm_fs };

	m_csm_program = GlobalGraphicsResources::load_program(vs_path + fs_path, 2, &shaders[0]);

	if (!m_csm_vs || !m_csm_fs || !m_csm_program)
	{
		DW_LOG_INFO("Failed to load Composition pass shaders");
	}

	m_csm_program->uniform_block_binding("u_PerFrame", 0);
	m_csm_program->uniform_block_binding("u_PerEntity", 1);
}

// -----------------------------------------------------------------------------------------------------------------------------------

ShadowMapRenderer::~ShadowMapRenderer() {}

// -----------------------------------------------------------------------------------------------------------------------------------

void ShadowMapRenderer::render(Scene* scene, CSM* csm_technique)
{
	// Bind states.
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);

	// Bind shader program.
	m_csm_program->use();

	for (int i = 0; i < csm_technique->frustum_split_count(); i++)
	{
		// Update global uniforms.
		m_csm_program->set_uniform("u_ViewProjection", csm_technique->split_view_proj(i));

		// Draw entire scene into frustum split framebuffer without materials.
		m_scene_renderer.render(scene, csm_technique->framebuffers()[i], m_csm_program, csm_technique->shadow_map_size(), csm_technique->shadow_map_size(), GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT, (float*)g_default_clear_color, 0, nullptr);
	}
}

// -----------------------------------------------------------------------------------------------------------------------------------