#pragma once

#include "post_process_renderer.h"

namespace nimble
{
	class Scene;

	class DeferredShadingRenderer
	{
	public:
		DeferredShadingRenderer();
		~DeferredShadingRenderer();
		void initialize(uint16_t width, uint16_t height);
		void shutdown();
		void profiling_gui();
		void on_window_resized(uint16_t width, uint16_t height);
		void render(Scene* scene, uint32_t w, uint32_t h);

	private:
		Texture* m_deferred_color;
		Texture* m_bright_pass;
		Framebuffer* m_deferred_fbo;

		Shader* m_deferred_vs;
		Shader* m_deferred_fs;
		Program* m_deferred_program;

		PostProcessRenderer m_post_process_renderer;
	};
}