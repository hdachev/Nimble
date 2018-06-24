#pragma once

#include "post_process_renderer.h"

class Scene;

class DeferredShadingRenderer
{
public:
	DeferredShadingRenderer();
	~DeferredShadingRenderer();
	void initialize(uint16_t width, uint16_t height);
	void shutdown();
	void on_window_resized(uint16_t width, uint16_t height);
	void render(Scene* scene, uint32_t w, uint32_t h);

private:
	dw::Texture* m_deferred_color;
	dw::Framebuffer* m_deferred_fbo;

	dw::Shader* m_deferred_vs;
	dw::Shader* m_deferred_fs;
	dw::Program* m_deferred_program;

	PostProcessRenderer m_post_process_renderer;
};