#include "final_composition.h"
#include "global_graphics_resources.h"
#include "logger.h"
#include "constants.h"

// -----------------------------------------------------------------------------------------------------------------------------------

FinalComposition::FinalComposition()
{
	std::string vs_path = "shader/post_process/quad_vs.glsl";
	m_composition_vs = GlobalGraphicsResources::load_shader(GL_VERTEX_SHADER, vs_path);

	std::string fs_path = "shader/post_process/final_composition/final_composition_fs.glsl";
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
		GlobalGraphicsResources::lookup_texture(RENDER_TARGET_TONE_MAPPING)->bind(0);

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
    
    if (m_composition_program->set_uniform("s_GBufferRT3", 6))
        GlobalGraphicsResources::lookup_texture(RENDER_TARGET_GBUFFER_RT3)->bind(6);

	if (m_composition_program->set_uniform("s_GBufferRTDepth", 7))
		GlobalGraphicsResources::lookup_texture(RENDER_TARGET_GBUFFER_DEPTH)->bind(7);

	if (m_composition_program->set_uniform("s_DeferredColor", 8))
		GlobalGraphicsResources::lookup_texture(RENDER_TARGET_TONE_MAPPING)->bind(8);

	if (m_composition_program->set_uniform("s_SSAO", 9))
		GlobalGraphicsResources::lookup_texture(RENDER_TARGET_SSAO)->bind(9);

	if (m_composition_program->set_uniform("s_SSAO_Blur", 10))
		GlobalGraphicsResources::lookup_texture(RENDER_TARGET_SSAO_BLUR)->bind(10);

	if (m_composition_program->set_uniform("s_BrightPass", 11))
		GlobalGraphicsResources::lookup_texture(RENDER_TARGET_BRIGHT_PASS)->bind(11);

	if (m_composition_program->set_uniform("s_SSR", 12))
		GlobalGraphicsResources::lookup_texture(RENDER_TARGET_SSR)->bind(12);

	m_post_process_renderer.render(w, h, nullptr);
}

// -----------------------------------------------------------------------------------------------------------------------------------
