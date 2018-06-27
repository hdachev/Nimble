#include "final_composition.h"
#include "global_graphics_resources.h"
#include "logger.h"
#include "constants.h"

// -----------------------------------------------------------------------------------------------------------------------------------

FinalComposition::FinalComposition()
{
	std::string vs_path = "shader/post_process/quad_vs.glsl";
	m_composition_vs = GlobalGraphicsResources::load_shader(GL_VERTEX_SHADER, vs_path);

	std::string fs_path = "shader/post_process/quad_fs.glsl";
	m_composition_fs = GlobalGraphicsResources::load_shader(GL_FRAGMENT_SHADER, fs_path);

	dw::Shader* shaders[] = { m_composition_vs, m_composition_fs };
    std::string combined_path = vs_path + fs_path;
	m_composition_program = GlobalGraphicsResources::load_program(combined_path, 2, &shaders[0]);

	if (!m_composition_vs || !m_composition_fs || !m_composition_program)
	{
		DW_LOG_INFO("Failed to load Composition pass shaders");
	}

	m_composition_program->uniform_block_binding("u_PerFrame", 0);
}

// -----------------------------------------------------------------------------------------------------------------------------------

FinalComposition::~FinalComposition() {}

// -----------------------------------------------------------------------------------------------------------------------------------

void FinalComposition::render(dw::Camera* camera, uint32_t w, uint32_t h)
{
	m_composition_program->use();

	GlobalGraphicsResources::per_frame_ubo()->bind_base(0);

	m_composition_program->set_uniform("u_NearPlane", camera->m_near);
	m_composition_program->set_uniform("u_FarPlane", camera->m_far);

	if (m_composition_program->set_uniform("s_Color", 0))
		GlobalGraphicsResources::lookup_texture(RENDER_TARGET_FORWARD_COLOR)->bind(0);

	if (m_composition_program->set_uniform("s_Depth", 1))
		GlobalGraphicsResources::lookup_texture(RENDER_TARGET_FORWARD_DEPTH)->bind(1);

	if (m_composition_program->set_uniform("s_CSMShadowMaps", 2))
		GlobalGraphicsResources::lookup_texture(CSM_SHADOW_MAPS)->bind(2);

	if (m_composition_program->set_uniform("s_GBufferRT0", 3))
		GlobalGraphicsResources::lookup_texture(RENDER_TARGET_GBUFFER_RT0)->bind(3);

	if (m_composition_program->set_uniform("s_GBufferRT1", 4))
		GlobalGraphicsResources::lookup_texture(RENDER_TARGET_GBUFFER_RT1)->bind(4);

	if (m_composition_program->set_uniform("s_GBufferRT2", 5))
		GlobalGraphicsResources::lookup_texture(RENDER_TARGET_GBUFFER_RT2)->bind(5);

	if (m_composition_program->set_uniform("s_GBufferRTDepth", 6))
		GlobalGraphicsResources::lookup_texture(RENDER_TARGET_GBUFFER_DEPTH)->bind(6);

	if (m_composition_program->set_uniform("s_DeferredColor", 7))
		GlobalGraphicsResources::lookup_texture(RENDER_TARGET_DEFERRED_COLOR)->bind(7);

	m_post_process_renderer.render(w, h, nullptr);
}

// -----------------------------------------------------------------------------------------------------------------------------------
