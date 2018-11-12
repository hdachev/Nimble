#pragma once

#include "post_process_renderer.h"

namespace nimble
{
	class MotionBlur
	{
	public:
		MotionBlur();
		~MotionBlur();
		void initialize(uint16_t width, uint16_t height);
		void shutdown();
		void profiling_gui();
		void on_window_resized(uint16_t width, uint16_t height);
		void render(uint32_t w, uint32_t h);

	private:
		PostProcessRenderer m_post_process_renderer;
		Texture2D* m_motion_blur_rt;
		Framebuffer* m_motion_blur_fbo;
		Shader* m_motion_blur_vs;
		Shader* m_motion_blur_fs;
		Program* m_motion_blur_program;
	};
}