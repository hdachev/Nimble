#pragma once

#include "post_process_renderer.h"

namespace nimble
{
	class DepthOfField
	{
	public:
		DepthOfField();
		~DepthOfField();
		void initialize(uint16_t width, uint16_t height);
		void shutdown();
		void profiling_gui();
		void on_window_resized(uint16_t width, uint16_t height);
		void render(uint32_t w, uint32_t h);

	private:
		void coc_generation(uint32_t w, uint32_t h);
		void downsample(uint32_t w, uint32_t h);
		void near_coc_max(uint32_t w, uint32_t h);
		void near_coc_blur(uint32_t w, uint32_t h);
		void dof_computation(uint32_t w, uint32_t h);
		void fill(uint32_t w, uint32_t h);
		void composite(uint32_t w, uint32_t h);

	public:
		float m_near_begin = 0.0f;
		float m_near_end = 0.0f;
		float m_far_begin = 200.0f;
		float m_far_end = 250.0f;
		float m_blend = 1.0f;
		float m_kernel_size = 1.0f;

	private:
		// Properties
		glm::vec2 m_kernel_scale;

		// CoC Pass
		Texture* m_coc_rt;
		Framebuffer* m_coc_fbo;

		Shader*  m_coc_fs;
		Program* m_coc_program;

		// Downsample Pass
		Texture* m_color4_rt;
		Texture* m_mul_coc_far4_rt;
		Texture* m_coc4_rt;
		Framebuffer* m_downsample_fbo;

		Shader*  m_downsample_fs;
		Program* m_downsample_program;

		// Near CoC Max X Pass
		Texture* m_near_coc_max_x4_rt;
		Framebuffer* m_near_coc_max_x_fbo;

		Shader*  m_near_coc_max_x4_fs;
		Program* m_near_coc_max_x_program;

		// Near CoC Max Pass
		Texture* m_near_coc_max4_rt;
		Framebuffer* m_near_coc_max_fbo;

		Shader*  m_near_coc_max4_fs;
		Program* m_near_coc_max_program;

		// Near CoC Blur X Pass
		Texture* m_near_coc_blur_x4_rt;
		Framebuffer* m_near_coc_blur_x_fbo;

		Shader*  m_near_coc_blur_x4_fs;
		Program* m_near_coc_blur_x_program;

		// Near CoC Blur Pass
		Texture* m_near_coc_blur4_rt;
		Framebuffer* m_near_coc_blur_fbo;

		Shader*  m_near_coc_blur4_fs;
		Program* m_near_coc_blur_program;

		// DoF Computation Pass
		Texture* m_near_dof4_rt;
		Texture* m_far_dof4_rt;
		Framebuffer* m_computation_fbo;

		Shader*  m_computation_fs;
		Program* m_computation_program;

		// Fill Pass
		Texture* m_near_fill_dof4_rt;
		Texture* m_far_fill_dof4_rt;
		Framebuffer* m_fill_fbo;

		Shader*  m_fill_fs;
		Program* m_fill_program;

		// Composite Pass
		Texture* m_composite_rt;
		Framebuffer* m_composite_fbo;

		Shader*  m_composite_fs;
		Program* m_composite_program;

		// Common VS
		Shader*  m_quad_vs;

		PostProcessRenderer m_post_process_renderer;
	};
}