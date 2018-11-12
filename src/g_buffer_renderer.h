#pragma once

#include "scene_renderer.h"

namespace nimble
{
	class GBufferRenderer
	{
	public:
		GBufferRenderer();
		~GBufferRenderer();
		void initialize(uint16_t width, uint16_t height);
		void shutdown();
		void profiling_gui();
		void on_window_resized(uint16_t width, uint16_t height);
		void render(Scene* scene, uint32_t w, uint32_t h);

	private:
		// Scene renderer for rendering geometry.
		SceneRenderer m_scene_renderer;

		// Temp: Global shaders and program.
		Shader* m_gbuffer_vs;
		Shader* m_gbuffer_fs;
		Program* m_gbuffer_program;

		// G-Buffer.
		Texture* m_gbuffer_rt0; // Albedo.rgb
		Texture* m_gbuffer_rt1; // -, -, Motion.x, Motion.y
		Texture* m_gbuffer_rt2; // Normal.x, Normal.y, Normal.z, -
		Texture* m_gbuffer_rt3; // Metalness, Roughness, Emissive Mask, Height
		Texture* m_gbuffer_depth;
		Framebuffer* m_gbuffer_fbo;

		// Queries
		Query m_queries[3];
		uint32_t m_query_index = 0;
		float m_last_result = 0;
	};
}
