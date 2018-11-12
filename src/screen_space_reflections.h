#pragma once

#include "post_process_renderer.h"

namespace nimble
{
	class ScreenSpaceReflections
	{
	public:
		ScreenSpaceReflections();
		~ScreenSpaceReflections();
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

		Texture* m_ssr_rt;
		Framebuffer* m_ssr_fbo;

		Shader* m_ssr_vs;
		Shader* m_ssr_fs;
		Program* m_ssr_program;

		PostProcessRenderer m_post_process_renderer;
	};
}