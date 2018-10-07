#include "deferred_shading_renderer.h"
#include "global_graphics_resources.h"
#include "constants.h"
#include "logger.h"
#include "scene.h"
#include "gpu_profiler.h"
#include <imgui.h>

// -----------------------------------------------------------------------------------------------------------------------------------

DeferredShadingRenderer::DeferredShadingRenderer() {}

// -----------------------------------------------------------------------------------------------------------------------------------

DeferredShadingRenderer::~DeferredShadingRenderer() {}

// -----------------------------------------------------------------------------------------------------------------------------------

void DeferredShadingRenderer::initialize(uint16_t width, uint16_t height)
{
	on_window_resized(width, height);

	std::string vs_path = "shader/post_process/quad_vs.glsl";
	m_deferred_vs = GlobalGraphicsResources::load_shader(GL_VERTEX_SHADER, vs_path);

	std::string fs_path = "shader/deferred/deferred_fs.glsl";
	m_deferred_fs = GlobalGraphicsResources::load_shader(GL_FRAGMENT_SHADER, fs_path);

	dw::Shader* shaders[] = { m_deferred_vs, m_deferred_fs };
    std::string combined_path = vs_path + fs_path;
	m_deferred_program = GlobalGraphicsResources::load_program(combined_path, 2, &shaders[0]);

	if (!m_deferred_vs || !m_deferred_fs || !m_deferred_program)
	{
		DW_LOG_ERROR("Failed to load G-Buffer pass shaders");
	}

	m_deferred_program->uniform_block_binding("u_PerFrame", 0);
	m_deferred_program->uniform_block_binding("u_PerScene", 1);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DeferredShadingRenderer::shutdown() {}

// -----------------------------------------------------------------------------------------------------------------------------------

void DeferredShadingRenderer::profiling_gui()
{
	ImGui::Text("Deferred Shading: %f ms", GPUProfiler::result("Deferred"));
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DeferredShadingRenderer::on_window_resized(uint16_t width, uint16_t height)
{
	// Clear earlier render targets.
	GlobalGraphicsResources::destroy_framebuffer(FRAMEBUFFER_DEFERRED);
	GlobalGraphicsResources::destroy_texture(RENDER_TARGET_DEFERRED_COLOR);

	// Create Render targets.
	m_deferred_color = GlobalGraphicsResources::create_texture_2d(RENDER_TARGET_DEFERRED_COLOR, width, height, GL_RGB32F, GL_RGB, GL_FLOAT);
	m_deferred_color->set_min_filter(GL_LINEAR);
	m_deferred_color->set_wrapping(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);

	// Create FBO.
	m_deferred_fbo = GlobalGraphicsResources::create_framebuffer(FRAMEBUFFER_DEFERRED);

	// Attach render target to FBO.
	m_deferred_fbo->attach_render_target(0, m_deferred_color, 0, 0);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DeferredShadingRenderer::render(Scene* scene, uint32_t w, uint32_t h)
{
	GPUProfiler::begin("Deferred");

	m_deferred_program->use();

	// Bind global UBO's.
	GlobalGraphicsResources::per_frame_ubo()->bind_base(0);
	GlobalGraphicsResources::per_scene_ubo()->bind_base(1);

	// Bind Textures.
	if (m_deferred_program->set_uniform("s_GBufferRT0", 0))
		GlobalGraphicsResources::lookup_texture(RENDER_TARGET_GBUFFER_RT0)->bind(0);

	if (m_deferred_program->set_uniform("s_GBufferRT1", 1))
		GlobalGraphicsResources::lookup_texture(RENDER_TARGET_GBUFFER_RT1)->bind(1);

	if (m_deferred_program->set_uniform("s_GBufferRT2", 2))
		GlobalGraphicsResources::lookup_texture(RENDER_TARGET_GBUFFER_RT2)->bind(2);
    
    if (m_deferred_program->set_uniform("s_GBufferRT3", 3))
        GlobalGraphicsResources::lookup_texture(RENDER_TARGET_GBUFFER_RT3)->bind(3);

	if (m_deferred_program->set_uniform("s_GBufferRTDepth", 4))
		GlobalGraphicsResources::lookup_texture(RENDER_TARGET_GBUFFER_DEPTH)->bind(4);

	if (m_deferred_program->set_uniform("s_ShadowMap", 5))
		GlobalGraphicsResources::lookup_texture(CSM_SHADOW_MAPS)->bind(5);

	if (m_deferred_program->set_uniform("s_IrradianceMap", 6))
		scene->irradiance_map()->bind(6);

	if (m_deferred_program->set_uniform("s_PrefilteredMap", 7))
		scene->prefiltered_map()->bind(7);

	if (m_deferred_program->set_uniform("s_BRDF",8))
	{
		dw::Texture* brdf_lut = GlobalGraphicsResources::lookup_texture(BRDF_LUT);
		brdf_lut->bind(8);
	}

	if (m_deferred_program->set_uniform("s_SSAO", 9))
		GlobalGraphicsResources::lookup_texture(RENDER_TARGET_SSAO_BLUR)->bind(9);

	if (m_deferred_program->set_uniform("s_SSR", 10))
		GlobalGraphicsResources::lookup_texture(RENDER_TARGET_SSR)->bind(10);

	m_post_process_renderer.render(w, h, m_deferred_fbo);

	GPUProfiler::end("Deferred");
}

// -----------------------------------------------------------------------------------------------------------------------------------
