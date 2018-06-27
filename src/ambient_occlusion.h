#pragma once

#include "post_process_renderer.h"

class AmbientOcclusion
{
public:
	AmbientOcclusion();
	~AmbientOcclusion();
	void initialize(uint16_t width, uint16_t height);
	void shutdown();
	void on_window_resized(uint16_t width, uint16_t height);
	void render(uint32_t w, uint32_t h);

private:
	// Render targets
	dw::Texture2D* m_ssao_rt;
	dw::Texture2D* m_ssao_vblur_rt;
	dw::Texture2D* m_ssao_hblur_rt;

	// Framebuffers
	dw::Framebuffer* m_ssao_fbo;
	dw::Framebuffer* m_ssao_vblur_fbo;
	dw::Framebuffer* m_ssao_hblur_fbo;

	// Shaders and programs
	dw::Shader* m_ssao_vs;
	dw::Shader* m_ssao_fs;
	dw::Shader* m_ssao_blur_vs;
	dw::Shader* m_ssao_blur_fs;
	dw::Program* m_ssao_program;
	dw::Program* m_ssao_blur_program;

	PostProcessRenderer m_post_process_renderer;
};