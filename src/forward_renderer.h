#pragma once

#include "scene_renderer.h"

namespace nimble
{
	class Scene;

	class ForwardRenderer
	{
	public:
		ForwardRenderer();
		~ForwardRenderer();
		void initialize(uint16_t width, uint16_t height);
		void shutdown();
		void profiling_gui();
		void on_window_resized(uint16_t width, uint16_t height);
		void render(Scene* scene, uint32_t w, uint32_t h);

	private:
		// Render targets
		Texture2D*	 m_color_buffer = nullptr;
		Texture2D*	 m_velocity_buffer = nullptr;
		Texture2D*	 m_bright_pass_buffer = nullptr;
		Texture2D*	 m_depth_buffer = nullptr;
						
		// Framebuffers	 
		Framebuffer* m_color_fbo = nullptr;

		// Renderers
		SceneRenderer m_scene_renderer;
	};
}