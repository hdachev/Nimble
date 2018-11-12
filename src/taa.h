#pragma once

#include "post_process_renderer.h"

namespace nimble
{
	class TAA
	{
	public:
		TAA();
		~TAA();
		void initialize(uint16_t width, uint16_t height);
		void shutdown();
		void profiling_gui();
		void on_window_resized(uint16_t width, uint16_t height);
		void render(uint32_t w, uint32_t h);

		inline bool is_enabled() { return m_enabled; }
		inline void enable() { m_enabled = true; }
		inline void disable() { m_enabled = false; }

	private:
		bool m_enabled;
		int32_t m_first;

		Texture* m_taa_rt;
		Framebuffer* m_taa_fbo;

		Texture* m_taa_hist_rt;
		Framebuffer* m_taa_hist_fbo;

		Shader*  m_quad_vs;

		// TAA shader
		Shader*  m_taa_fs;
		Program* m_taa_program;

		Shader*  m_taa_hist_fs;
		Program* m_taa_hist_program;

		PostProcessRenderer m_post_process_renderer;
	};
}