#include "depth_of_field.h"
#include "global_graphics_resources.h"
#include "constants.h"
#include "logger.h"
#include "gpu_profiler.h"
#include "imgui.h"

namespace nimble
{
	// -----------------------------------------------------------------------------------------------------------------------------------

	DepthOfField::DepthOfField()
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	DepthOfField::~DepthOfField()
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void DepthOfField::initialize(uint16_t width, uint16_t height)
	{
		on_window_resized(width, height);

		std::string vs_path = "shader/post_process/quad_vs.glsl";
		m_quad_vs = GlobalGraphicsResources::load_shader(GL_VERTEX_SHADER, vs_path);

		{
			std::string fs_path = "shader/post_process/depth_of_field/coc_fs.glsl";
			m_coc_fs = GlobalGraphicsResources::load_shader(GL_FRAGMENT_SHADER, fs_path);

			Shader* shaders[] = { m_quad_vs, m_coc_fs };
			std::string combined_path = vs_path + fs_path;
			m_coc_program = GlobalGraphicsResources::load_program(combined_path, 2, &shaders[0]);

			if (!m_quad_vs || !m_coc_fs || !m_coc_program)
			{
				NIMBLE_LOG_ERROR("Failed to load DOF CoC shaders");
			}

			m_coc_program->uniform_block_binding("u_PerFrame", 0);
		}

		{
			std::string fs_path = "shader/post_process/depth_of_field/composite_fs.glsl";
			m_composite_fs = GlobalGraphicsResources::load_shader(GL_FRAGMENT_SHADER, fs_path);

			Shader* shaders[] = { m_quad_vs, m_composite_fs };
			std::string combined_path = vs_path + fs_path;
			m_composite_program = GlobalGraphicsResources::load_program(combined_path, 2, &shaders[0]);

			if (!m_quad_vs || !m_composite_fs || !m_composite_program)
			{
				NIMBLE_LOG_ERROR("Failed to load DOF Composite shaders");
			}
		}

		{
			std::string fs_path = "shader/post_process/depth_of_field/computation_fs.glsl";
			m_computation_fs = GlobalGraphicsResources::load_shader(GL_FRAGMENT_SHADER, fs_path);

			Shader* shaders[] = { m_quad_vs, m_computation_fs };
			std::string combined_path = vs_path + fs_path;
			m_computation_program = GlobalGraphicsResources::load_program(combined_path, 2, &shaders[0]);

			if (!m_quad_vs || !m_computation_fs || !m_computation_program)
			{
				NIMBLE_LOG_ERROR("Failed to load DOF Computation shaders");
			}
		}

		{
			std::string fs_path = "shader/post_process/depth_of_field/downsample_fs.glsl";
			m_downsample_fs = GlobalGraphicsResources::load_shader(GL_FRAGMENT_SHADER, fs_path);

			Shader* shaders[] = { m_quad_vs, m_downsample_fs };
			std::string combined_path = vs_path + fs_path;
			m_downsample_program = GlobalGraphicsResources::load_program(combined_path, 2, &shaders[0]);

			if (!m_quad_vs || !m_downsample_fs || !m_downsample_program)
			{
				NIMBLE_LOG_ERROR("Failed to load DOF Downsample shaders");
			}
		}

		{
			std::string fs_path = "shader/post_process/depth_of_field/fill_fs.glsl";
			m_fill_fs = GlobalGraphicsResources::load_shader(GL_FRAGMENT_SHADER, fs_path);

			Shader* shaders[] = { m_quad_vs, m_fill_fs };
			std::string combined_path = vs_path + fs_path;
			m_fill_program = GlobalGraphicsResources::load_program(combined_path, 2, &shaders[0]);

			if (!m_quad_vs || !m_fill_fs || !m_fill_program)
			{
				NIMBLE_LOG_ERROR("Failed to load DOF Fill shaders");
			}
		}

		{
			std::string fs_path = "shader/post_process/filters_fs.glsl";
			
			// Near CoC Max X
			{
				m_near_coc_max_x4_fs = GlobalGraphicsResources::load_shader(GL_FRAGMENT_SHADER, fs_path, { "HORIZONTAL", "MAX13", "CHANNELS_COUNT_1" });

				Shader* shaders[] = { m_quad_vs, m_near_coc_max_x4_fs };
				std::string combined_name = GlobalGraphicsResources::combined_program_name(vs_path, fs_path, { "HORIZONTAL", "MAX13", "CHANNELS_COUNT_1" });
				m_near_coc_max_x_program = GlobalGraphicsResources::load_program(combined_name, 2, &shaders[0]);

				if (!m_quad_vs || !m_near_coc_max_x4_fs || !m_near_coc_max_x_program)
				{
					NIMBLE_LOG_ERROR("Failed to load Near CoC Max X shaders");
				}
			}

			// Near CoC Max
			{
				m_near_coc_max4_fs = GlobalGraphicsResources::load_shader(GL_FRAGMENT_SHADER, fs_path, { "VERTICAL", "MAX13", "CHANNELS_COUNT_1" });

				Shader* shaders[] = { m_quad_vs, m_near_coc_max4_fs };
				std::string combined_name = GlobalGraphicsResources::combined_program_name(vs_path, fs_path, { "VERTICAL", "MAX13", "CHANNELS_COUNT_1" });
				m_near_coc_max_program = GlobalGraphicsResources::load_program(combined_name, 2, &shaders[0]);

				if (!m_quad_vs || !m_near_coc_max4_fs || !m_near_coc_max_program)
				{
					NIMBLE_LOG_ERROR("Failed to load Near CoC Max shaders");
				}
			}

			// Near CoC Blur X
			{
				m_near_coc_blur_x4_fs = GlobalGraphicsResources::load_shader(GL_FRAGMENT_SHADER, fs_path, { "HORIZONTAL", "BLUR13", "CHANNELS_COUNT_1" });

				Shader* shaders[] = { m_quad_vs, m_near_coc_blur_x4_fs };
				std::string combined_name = GlobalGraphicsResources::combined_program_name(vs_path, fs_path, { "HORIZONTAL", "BLUR13", "CHANNELS_COUNT_1" });
				m_near_coc_blur_x_program = GlobalGraphicsResources::load_program(combined_name, 2, &shaders[0]);

				if (!m_quad_vs || !m_near_coc_blur_x4_fs || !m_near_coc_blur_x_program)
				{
					NIMBLE_LOG_ERROR("Failed to load Near CoC Blur X shaders");
				}
			}

			// Near CoC Blur
			{
				m_near_coc_blur4_fs = GlobalGraphicsResources::load_shader(GL_FRAGMENT_SHADER, fs_path, { "VERTICAL", "BLUR13", "CHANNELS_COUNT_1" });

				Shader* shaders[] = { m_quad_vs, m_near_coc_blur4_fs };
				std::string combined_name = GlobalGraphicsResources::combined_program_name(vs_path, fs_path, { "VERTICAL", "BLUR13", "CHANNELS_COUNT_1" });
				m_near_coc_blur_program = GlobalGraphicsResources::load_program(combined_name, 2, &shaders[0]);

				if (!m_quad_vs || !m_near_coc_blur4_fs || !m_near_coc_blur_program)
				{
					NIMBLE_LOG_ERROR("Failed to load Near CoC Blur shaders");
				}
			}
		}
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void DepthOfField::shutdown()
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void DepthOfField::profiling_gui()
	{
		ImGui::Text("DOF - CoC: %f ms", GPUProfiler::result("DOF - CoC"));
		ImGui::Text("DOF - Downsample: %f ms", GPUProfiler::result("DOF - Downsample"));
		ImGui::Text("DOF - Near Max: %f ms", GPUProfiler::result("DOF - Near CoC Max"));
		ImGui::Text("DOF - Near Blur: %f ms", GPUProfiler::result("DOF - Near CoC Blur"));
		ImGui::Text("DOF - Computation: %f ms", GPUProfiler::result("DOF - Computation"));
		ImGui::Text("DOF - Fill: %f ms", GPUProfiler::result("DOF - Fill"));
		ImGui::Text("DOF - Composite: %f ms", GPUProfiler::result("DOF - Composite"));
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void DepthOfField::on_window_resized(uint16_t width, uint16_t height)
	{
		// CoC
		GlobalGraphicsResources::destroy_framebuffer(FRAMEBUFFER_DOF_COC);
		GlobalGraphicsResources::destroy_texture(RENDER_TARGET_DOF_COC);

		m_coc_rt = GlobalGraphicsResources::create_texture_2d(RENDER_TARGET_DOF_COC, width, height, GL_RG8, GL_RG, GL_UNSIGNED_BYTE);
		m_coc_rt->set_min_filter(GL_NEAREST);
		m_coc_rt->set_wrapping(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);

		m_coc_fbo = GlobalGraphicsResources::create_framebuffer(FRAMEBUFFER_DOF_COC);
		m_coc_fbo->attach_render_target(0, m_coc_rt, 0, 0);

		// Downsample
		GlobalGraphicsResources::destroy_framebuffer(FRAMEBUFFER_DOF_DOWNSAMPLE);
		GlobalGraphicsResources::destroy_texture(RENDER_TARGET_DOF_COLOR4);
		GlobalGraphicsResources::destroy_texture(RENDER_TARGET_DOF_COC4);
		GlobalGraphicsResources::destroy_texture(RENDER_TARGET_DOF_MUL_COC_FAR4);

		m_color4_rt = GlobalGraphicsResources::create_texture_2d(RENDER_TARGET_DOF_COLOR4, width / 2, height / 2, GL_RGB32F, GL_RGB, GL_FLOAT);
		m_color4_rt->set_min_filter(GL_LINEAR);
		m_color4_rt->set_wrapping(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);

		m_mul_coc_far4_rt = GlobalGraphicsResources::create_texture_2d(RENDER_TARGET_DOF_MUL_COC_FAR4, width / 2, height / 2, GL_RGB32F, GL_RGB, GL_FLOAT);
		m_mul_coc_far4_rt->set_min_filter(GL_LINEAR);
		m_mul_coc_far4_rt->set_wrapping(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);

		m_coc4_rt = GlobalGraphicsResources::create_texture_2d(RENDER_TARGET_DOF_COC4, width / 2, height / 2, GL_RG8, GL_RG, GL_UNSIGNED_BYTE);
		m_coc4_rt->set_min_filter(GL_LINEAR);
		m_coc4_rt->set_wrapping(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);

		Texture* ds_render_targets[] = { m_color4_rt, m_mul_coc_far4_rt, m_coc4_rt };
		m_downsample_fbo = GlobalGraphicsResources::create_framebuffer(FRAMEBUFFER_DOF_DOWNSAMPLE);
		m_downsample_fbo->attach_multiple_render_targets(3, ds_render_targets);

		// Near CoC Max X
		GlobalGraphicsResources::destroy_framebuffer(FRAMEBUFFER_DOF_NEAR_COC_MAX_X4);
		GlobalGraphicsResources::destroy_texture(RENDER_TARGET_NEAR_COC_MAX_X4);

		m_near_coc_max_x4_rt = GlobalGraphicsResources::create_texture_2d(RENDER_TARGET_NEAR_COC_MAX_X4, width / 2, height / 2, GL_R8, GL_RED, GL_UNSIGNED_BYTE);
		m_near_coc_max_x4_rt->set_min_filter(GL_NEAREST);
		m_near_coc_max_x4_rt->set_wrapping(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);

		m_near_coc_max_x_fbo = GlobalGraphicsResources::create_framebuffer(FRAMEBUFFER_DOF_NEAR_COC_MAX_X4);
		m_near_coc_max_x_fbo->attach_render_target(0, m_near_coc_max_x4_rt, 0, 0);

		// Near CoC Max
		GlobalGraphicsResources::destroy_framebuffer(FRAMEBUFFER_DOF_NEAR_COC_MAX4);
		GlobalGraphicsResources::destroy_texture(RENDER_TARGET_NEAR_COC_MAX4);

		m_near_coc_max4_rt = GlobalGraphicsResources::create_texture_2d(RENDER_TARGET_NEAR_COC_MAX4, width / 2, height / 2, GL_R8, GL_RED, GL_UNSIGNED_BYTE);
		m_near_coc_max4_rt->set_min_filter(GL_NEAREST);
		m_near_coc_max4_rt->set_wrapping(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);

		m_near_coc_max_fbo = GlobalGraphicsResources::create_framebuffer(FRAMEBUFFER_DOF_NEAR_COC_MAX4);
		m_near_coc_max_fbo->attach_render_target(0, m_near_coc_max4_rt, 0, 0);

		// Near CoC Blur X
		GlobalGraphicsResources::destroy_framebuffer(FRAMEBUFFER_NEAR_COC_BLUR_X4);
		GlobalGraphicsResources::destroy_texture(RENDER_TARGET_NEAR_COC_BLUR_X4);

		m_near_coc_blur_x4_rt = GlobalGraphicsResources::create_texture_2d(RENDER_TARGET_NEAR_COC_BLUR_X4, width / 2, height / 2, GL_R8, GL_RED, GL_UNSIGNED_BYTE);
		m_near_coc_blur_x4_rt->set_min_filter(GL_NEAREST);
		m_near_coc_blur_x4_rt->set_wrapping(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);
				
		m_near_coc_blur_x_fbo = GlobalGraphicsResources::create_framebuffer(FRAMEBUFFER_NEAR_COC_BLUR_X4);
		m_near_coc_blur_x_fbo->attach_render_target(0, m_near_coc_blur_x4_rt, 0, 0);

		// Near CoC Blur
		GlobalGraphicsResources::destroy_framebuffer(FRAMEBUFFER_NEAR_COC_BLUR4);
		GlobalGraphicsResources::destroy_texture(RENDER_TARGET_NEAR_COC_BLUR4);

		m_near_coc_blur4_rt = GlobalGraphicsResources::create_texture_2d(RENDER_TARGET_NEAR_COC_BLUR4, width / 2, height / 2, GL_R8, GL_RED, GL_UNSIGNED_BYTE);
		m_near_coc_blur4_rt->set_min_filter(GL_NEAREST);
		m_near_coc_blur4_rt->set_wrapping(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);
				
		m_near_coc_blur_fbo = GlobalGraphicsResources::create_framebuffer(FRAMEBUFFER_NEAR_COC_BLUR4);
		m_near_coc_blur_fbo->attach_render_target(0, m_near_coc_blur4_rt, 0, 0);

		// Computation
		GlobalGraphicsResources::destroy_framebuffer(FRAMEBUFFER_DOF4);
		GlobalGraphicsResources::destroy_texture(RENDER_TARGET_NEAR_DOF4);
		GlobalGraphicsResources::destroy_texture(RENDER_TARGET_FAR_DOF4);

		m_near_dof4_rt = GlobalGraphicsResources::create_texture_2d(RENDER_TARGET_NEAR_DOF4, width / 2, height / 2, GL_RGB32F, GL_RGB, GL_FLOAT);
		m_near_dof4_rt->set_min_filter(GL_LINEAR);
		m_near_dof4_rt->set_wrapping(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);

		m_far_dof4_rt = GlobalGraphicsResources::create_texture_2d(RENDER_TARGET_FAR_DOF4, width / 2, height / 2, GL_RGB32F, GL_RGB, GL_FLOAT);
		m_far_dof4_rt->set_min_filter(GL_LINEAR);
		m_far_dof4_rt->set_wrapping(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);

		Texture* comp_render_targets[] = { m_near_dof4_rt, m_far_dof4_rt };
		m_computation_fbo = GlobalGraphicsResources::create_framebuffer(FRAMEBUFFER_DOF4);
		m_computation_fbo->attach_multiple_render_targets(2, comp_render_targets);

		// Fill
		GlobalGraphicsResources::destroy_framebuffer(FRAMEBUFFER_DOF_FILL4);
		GlobalGraphicsResources::destroy_texture(RENDER_TARGET_NEAR_FILL_DOF4);
		GlobalGraphicsResources::destroy_texture(RENDER_TARGET_FAR_FILL_DOF4);

		m_near_fill_dof4_rt = GlobalGraphicsResources::create_texture_2d(RENDER_TARGET_NEAR_FILL_DOF4, width / 2, height / 2, GL_RGB32F, GL_RGB, GL_FLOAT);
		m_near_fill_dof4_rt->set_min_filter(GL_LINEAR);
		m_near_fill_dof4_rt->set_wrapping(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);

		m_far_fill_dof4_rt = GlobalGraphicsResources::create_texture_2d(RENDER_TARGET_FAR_FILL_DOF4, width / 2, height / 2, GL_RGB32F, GL_RGB, GL_FLOAT);
		m_far_fill_dof4_rt->set_min_filter(GL_LINEAR);
		m_far_fill_dof4_rt->set_wrapping(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);

		Texture* fill_render_targets[] = { m_near_fill_dof4_rt, m_far_fill_dof4_rt };
		m_fill_fbo = GlobalGraphicsResources::create_framebuffer(FRAMEBUFFER_DOF_FILL4);
		m_fill_fbo->attach_multiple_render_targets(2, fill_render_targets);

		// Composite
		GlobalGraphicsResources::destroy_framebuffer(FRAMEBUFFER_DOF_COMPOSITE);
		GlobalGraphicsResources::destroy_texture(RENDER_TARGET_DOF_COMPOSITE);

		m_composite_rt = GlobalGraphicsResources::create_texture_2d(RENDER_TARGET_DOF_COMPOSITE, width, height, GL_RGB32F, GL_RGB, GL_FLOAT);
		m_composite_rt->set_min_filter(GL_LINEAR);
		m_composite_rt->set_wrapping(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);

		m_composite_fbo = GlobalGraphicsResources::create_framebuffer(FRAMEBUFFER_DOF_COMPOSITE);
		m_composite_fbo->attach_render_target(0, m_composite_rt, 0, 0);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void DepthOfField::render(uint32_t w, uint32_t h)
	{
		coc_generation(w, h);
		downsample(w, h);
		near_coc_max(w, h);
		near_coc_blur(w, h);
		dof_computation(w, h);
		fill(w, h);
		composite(w, h);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void DepthOfField::coc_generation(uint32_t w, uint32_t h)
	{
		GPUProfiler::begin("DOF - CoC");

		m_coc_program->use();

		// Bind global UBO's.
		GlobalGraphicsResources::per_frame_ubo()->bind_base(0);

		glm::vec2 pixel_size = glm::vec2(1.0f / float(w), 1.0f / float(h));
		m_coc_program->set_uniform("u_PixelSize", pixel_size);
		m_coc_program->set_uniform("u_NearBegin", m_near_begin);
		m_coc_program->set_uniform("u_NearEnd", m_near_end);
		m_coc_program->set_uniform("u_FarBegin", m_far_begin);
		m_coc_program->set_uniform("u_FarEnd", m_far_end);

		if (m_coc_program->set_uniform("s_Depth", 0))
			GlobalGraphicsResources::lookup_texture(RENDER_TARGET_GBUFFER_DEPTH)->bind(0);

		m_post_process_renderer.render(w, h, m_coc_fbo);

		GPUProfiler::end("DOF - CoC");
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void DepthOfField::downsample(uint32_t w, uint32_t h)
	{
		GPUProfiler::begin("DOF - Downsample");

		m_downsample_program->use();

		// Bind global UBO's.
		GlobalGraphicsResources::per_frame_ubo()->bind_base(0);

		glm::vec2 pixel_size = glm::vec2(1.0f / float(w), 1.0f / float(h));
		m_downsample_program->set_uniform("u_PixelSize", pixel_size);

		PerFrameUniforms& per_frame = GlobalGraphicsResources::per_frame_uniforms();

		if (m_downsample_program->set_uniform("s_Color", 0))
			GlobalGraphicsResources::lookup_texture(RENDER_TARGET_TAA)->bind(0);

		if (m_downsample_program->set_uniform("s_CoC", 1))
			m_coc_rt->bind(1);

		m_post_process_renderer.render(w / 2, h / 2, m_downsample_fbo);

		GPUProfiler::end("DOF - Downsample");
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void DepthOfField::near_coc_max(uint32_t w, uint32_t h)
	{
		GPUProfiler::begin("DOF - Near CoC Max");

		// Horizontal
		m_near_coc_max_x_program->use();

		glm::vec2 pixel_size = glm::vec2(1.0f / float(w/2), 1.0f / float(h/2));
		m_near_coc_max_x_program->set_uniform("u_PixelSize", pixel_size);

		if (m_near_coc_max_x_program->set_uniform("s_Texture", 0))
			m_coc4_rt->bind(0);

		m_post_process_renderer.render(w / 2, h / 2, m_near_coc_max_x_fbo);

		// Vertical
		m_near_coc_max_program->use();

		m_near_coc_max_program->set_uniform("u_PixelSize", pixel_size);

		if (m_near_coc_max_program->set_uniform("s_Texture", 0))
			m_near_coc_max_x4_rt->bind(0);

		m_post_process_renderer.render(w / 2, h / 2, m_near_coc_max_fbo);

		GPUProfiler::end("DOF - Near CoC Max");
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void DepthOfField::near_coc_blur(uint32_t w, uint32_t h)
	{
		GPUProfiler::begin("DOF - Near CoC Blur");

		// Horizontal
		m_near_coc_blur_x_program->use();

		glm::vec2 pixel_size = glm::vec2(1.0f / float(w/2), 1.0f / float(h/2));
		m_near_coc_blur_x_program->set_uniform("u_PixelSize", pixel_size);

		if (m_near_coc_blur_x_program->set_uniform("s_Texture", 0))
			m_near_coc_max4_rt->bind(0);

		m_post_process_renderer.render(w / 2, h / 2, m_near_coc_blur_x_fbo);

		// Vertical
		m_near_coc_blur_program->use();

		m_near_coc_blur_program->set_uniform("u_PixelSize", pixel_size);

		if (m_near_coc_blur_program->set_uniform("s_Texture", 0))
			m_near_coc_blur_x4_rt->bind(0);

		m_post_process_renderer.render(w / 2, h / 2, m_near_coc_blur_fbo);

		GPUProfiler::end("DOF - Near CoC Blur");
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void DepthOfField::dof_computation(uint32_t w, uint32_t h)
	{
		GPUProfiler::begin("DOF - Computation");

		m_computation_program->use();

		glm::vec2 pixel_size = glm::vec2(1.0f / float(w/2), 1.0f / float(h/2));
		m_computation_program->set_uniform("u_PixelSize", pixel_size);
		m_computation_program->set_uniform("u_KernelSize", m_kernel_size);

		if (m_computation_program->set_uniform("s_CoC4", 0))
			m_coc4_rt->bind(0);
		
		if (m_computation_program->set_uniform("s_NearBlurCoC4", 1))
			m_near_coc_blur4_rt->bind(1);

		if (m_computation_program->set_uniform("s_Color4", 2))
			m_color4_rt->bind(2);

		if (m_computation_program->set_uniform("s_ColorFarCoC4", 3))
			m_mul_coc_far4_rt->bind(3);

		m_post_process_renderer.render(w / 2, h / 2, m_computation_fbo);

		GPUProfiler::end("DOF - Computation");
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void DepthOfField::fill(uint32_t w, uint32_t h)
	{
		GPUProfiler::begin("DOF - Fill");

		m_fill_program->use();

		glm::vec2 pixel_size = glm::vec2(1.0f / float(w / 2), 1.0f / float(h / 2));
		m_fill_program->set_uniform("u_PixelSize", pixel_size);

		if (m_fill_program->set_uniform("s_CoC4", 0))
			m_coc4_rt->bind(0);

		if (m_fill_program->set_uniform("s_NearBlurCoC4", 1))
			m_near_coc_blur4_rt->bind(1);

		if (m_fill_program->set_uniform("s_NearDoF4", 2))
			m_near_dof4_rt->bind(2);

		if (m_fill_program->set_uniform("s_FarDoF4", 3))
			m_far_dof4_rt->bind(3);

		m_post_process_renderer.render(w / 2, h / 2, m_fill_fbo);

		GPUProfiler::end("DOF - Fill");
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void DepthOfField::composite(uint32_t w, uint32_t h)
	{
		GPUProfiler::begin("DOF - Composite");

		m_composite_program->use();

		glm::vec2 pixel_size = glm::vec2(1.0f / float(w / 2), 1.0f / float(h / 2));
		m_composite_program->set_uniform("u_PixelSize", pixel_size);
		m_composite_program->set_uniform("u_Blend", m_blend);

		if (m_composite_program->set_uniform("s_Color", 0))
			GlobalGraphicsResources::lookup_texture(RENDER_TARGET_TAA)->bind(0);

		if (m_composite_program->set_uniform("s_CoC", 1))
			m_coc_rt->bind(1);

		if (m_composite_program->set_uniform("s_CoC4", 2))
			m_coc4_rt->bind(2);

		if (m_composite_program->set_uniform("s_CoCBlur4", 3))
			m_near_coc_blur4_rt->bind(3);

		if (m_composite_program->set_uniform("s_NearDoF4", 4))
			m_near_fill_dof4_rt->bind(4);

		if (m_composite_program->set_uniform("s_FarDoF4", 5))
			m_far_fill_dof4_rt->bind(5);

		m_post_process_renderer.render(w, h, m_composite_fbo);

		GPUProfiler::end("DOF - Composite");
	}

	// -----------------------------------------------------------------------------------------------------------------------------------
}