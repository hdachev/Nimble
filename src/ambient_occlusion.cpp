#include "ambient_occlusion.h"
#include "global_graphics_resources.h"
#include "logger.h"
#include "constants.h"
#include "gpu_profiler.h"
#include "imgui.h"
#define GLM_ENABLE_EXPERIMENTAL
#include <gtx/compatibility.hpp>
#include <random>

namespace nimble
{
	// -----------------------------------------------------------------------------------------------------------------------------------

	AmbientOcclusion::AmbientOcclusion() {}

	// -----------------------------------------------------------------------------------------------------------------------------------

	AmbientOcclusion::~AmbientOcclusion() {}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void AmbientOcclusion::initialize(uint16_t width, uint16_t height)
	{
		on_window_resized(width, height);

		{
			std::string vs_path = "shader/post_process/quad_vs.glsl";
			m_ssao_vs = GlobalGraphicsResources::load_shader(GL_VERTEX_SHADER, vs_path);

			std::string fs_path = "shader/post_process/ssao/ssao_fs.glsl";
			m_ssao_fs = GlobalGraphicsResources::load_shader(GL_FRAGMENT_SHADER, fs_path);

			Shader* shaders[] = { m_ssao_vs, m_ssao_fs };
			std::string combined_path = vs_path + fs_path;
			m_ssao_program = GlobalGraphicsResources::load_program(combined_path, 2, &shaders[0]);

			if (!m_ssao_vs || !m_ssao_fs || !m_ssao_program)
			{
				NIMBLE_LOG_ERROR("Failed to load SSAO shaders");
			}

			m_ssao_program->uniform_block_binding("u_PerFrame", 0);
			m_ssao_program->uniform_block_binding("u_SSAOData", 1);
		}

		{
			std::string vs_path = "shader/post_process/quad_vs.glsl";
			m_ssao_blur_vs = GlobalGraphicsResources::load_shader(GL_VERTEX_SHADER, vs_path);

			std::string fs_path = "shader/post_process/ssao/ssao_blur_fs.glsl";
			m_ssao_blur_fs = GlobalGraphicsResources::load_shader(GL_FRAGMENT_SHADER, fs_path);

			Shader* shaders[] = { m_ssao_blur_vs, m_ssao_blur_fs };
			std::string combined_path = vs_path + fs_path;
			m_ssao_blur_program = GlobalGraphicsResources::load_program(combined_path, 2, &shaders[0]);

			if (!m_ssao_blur_vs || !m_ssao_blur_fs || !m_ssao_blur_program)
			{
				NIMBLE_LOG_ERROR("Failed to load SSAO blur shaders");
			}
		}

		std::uniform_real_distribution<float> random_floats(0.0, 1.0);
		std::default_random_engine generator;

		std::vector<glm::vec3> ssao_noise;

		for (uint32_t i = 0; i < 16; i++)
		{
			glm::vec3 noise(random_floats(generator) * 2.0f - 1.0f, random_floats(generator) * 2.0f - 1.0f, 0.0f);
			ssao_noise.push_back(noise);
		}

		m_noise_texture = std::make_unique<Texture2D>(4, 4, 1, 1, 1, GL_RGB16F, GL_RGB, GL_FLOAT);
		m_noise_texture->set_min_filter(GL_NEAREST);
		m_noise_texture->set_mag_filter(GL_NEAREST);
		m_noise_texture->set_wrapping(GL_REPEAT, GL_REPEAT, GL_REPEAT);
		m_noise_texture->set_data(0, 0, ssao_noise.data());

		std::vector<glm::vec4> ssao_kernel;

		for (uint32_t i = 0; i < 64; i++)
		{
			glm::vec4 sample = glm::vec4(random_floats(generator) * 2.0f - 1.0f, random_floats(generator) * 2.0f - 1.0f, random_floats(generator), 0.0f);
			sample = glm::normalize(sample);
			sample *= random_floats(generator);

			float scale = float(i) / 64.0f;
			scale = glm::lerp(0.1f, 1.0f, scale * scale);
			sample *= scale;

			ssao_kernel.push_back(sample);
		}

		m_kernel_ubo = std::make_unique<UniformBuffer>(GL_DYNAMIC_DRAW, sizeof(SSAOData), ssao_kernel.data());
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void AmbientOcclusion::shutdown() {}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void AmbientOcclusion::profiling_gui()
	{
		ImGui::Text("SSAO Buffer: %f ms", GPUProfiler::result("SSAOBuffer"));
		ImGui::Text("SSAO Blur: %f ms", GPUProfiler::result("SSAOBlur"));
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void AmbientOcclusion::on_window_resized(uint16_t width, uint16_t height)
	{
		// Clear earlier render targets.
		GlobalGraphicsResources::destroy_framebuffer(FRAMEBUFFER_SSAO);
		GlobalGraphicsResources::destroy_framebuffer(FRAMEBUFFER_SSAO_BLUR);
		GlobalGraphicsResources::destroy_texture(RENDER_TARGET_SSAO);
		GlobalGraphicsResources::destroy_texture(RENDER_TARGET_SSAO_BLUR);

		// Create Render targets.
		m_ssao_rt = GlobalGraphicsResources::create_texture_2d(RENDER_TARGET_SSAO, width / 2, height / 2, GL_R8, GL_RED, GL_UNSIGNED_BYTE);
		m_ssao_rt->set_min_filter(GL_LINEAR);

		m_ssao_blur_rt = GlobalGraphicsResources::create_texture_2d(RENDER_TARGET_SSAO_BLUR, width / 2, height / 2, GL_R8, GL_RED, GL_UNSIGNED_BYTE);
		m_ssao_blur_rt->set_min_filter(GL_LINEAR);

		// Create FBO.
		m_ssao_fbo = GlobalGraphicsResources::create_framebuffer(FRAMEBUFFER_SSAO);
		m_ssao_blur_fbo = GlobalGraphicsResources::create_framebuffer(FRAMEBUFFER_SSAO_BLUR);

		// Attach render target to FBO.
		m_ssao_fbo->attach_render_target(0, m_ssao_rt, 0, 0);
		m_ssao_blur_fbo->attach_render_target(0, m_ssao_blur_rt, 0, 0);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void AmbientOcclusion::render(uint32_t w, uint32_t h)
	{
		render_ssao(w / 2, h / 2);
		render_blur(w / 2, h / 2);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void AmbientOcclusion::render_ssao(uint32_t w, uint32_t h)
	{
		GPUProfiler::begin("SSAOBuffer");

		m_ssao_program->use();

		GlobalGraphicsResources::per_frame_ubo()->bind_base(0);
		m_kernel_ubo->bind_base(1);

		//PerFrameUniforms& per_frame = GlobalGraphicsResources::per_frame_uniforms();

		if (m_ssao_program->set_uniform("s_GBufferNormals", 0))
			GlobalGraphicsResources::lookup_texture(RENDER_TARGET_GBUFFER_RT2)->bind(0);

		if (m_ssao_program->set_uniform("s_GBufferRTDepth", 1))
			GlobalGraphicsResources::lookup_texture(RENDER_TARGET_GBUFFER_DEPTH)->bind(1);

		if (m_ssao_program->set_uniform("s_Noise", 2))
			m_noise_texture->bind(2);

		m_post_process_renderer.render(w, h, m_ssao_fbo);

		GPUProfiler::end("SSAOBuffer");
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void AmbientOcclusion::render_blur(uint32_t w, uint32_t h)
	{
		GPUProfiler::begin("SSAOBlur");

		m_ssao_blur_program->use();

		GlobalGraphicsResources::per_frame_ubo()->bind_base(0);

		if (m_ssao_blur_program->set_uniform("s_SSAO", 0))
			GlobalGraphicsResources::lookup_texture(RENDER_TARGET_SSAO)->bind(0);

		m_post_process_renderer.render(w, h, m_ssao_blur_fbo);

		GPUProfiler::end("SSAOBlur");
	}

	// -----------------------------------------------------------------------------------------------------------------------------------
}