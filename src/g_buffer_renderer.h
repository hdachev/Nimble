#pragma once

#include "scene_renderer.h"

class GBufferRenderer
{
public:
	GBufferRenderer();
	~GBufferRenderer();
	void initialize(uint16_t width, uint16_t height);
	void shutdown();
	void on_window_resized(uint16_t width, uint16_t height);
	void render(Scene* scene, uint32_t w, uint32_t h);

private:
	// Scene renderer for rendering geometry.
	SceneRenderer m_scene_renderer;

	// Temp: Global shaders and program.
	dw::Shader* m_gbuffer_vs;
	dw::Shader* m_gbuffer_fs;
	dw::Program* m_gbuffer_program;

	// G-Buffer.
	dw::Texture* m_gbuffer_rt0; // Albedo.rgb
	dw::Texture* m_gbuffer_rt1; // Normal.x, Normal.y, Motion.x, Motion.y
	dw::Texture* m_gbuffer_rt2; // Metalness, Roughness, Emissive Mask, Height
	dw::Texture* m_gbuffer_rt3; // Position (TEMP).
	dw::Texture* m_gbuffer_depth;
	dw::Framebuffer* m_gbuffer_fbo;
};