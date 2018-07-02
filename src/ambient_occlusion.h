#pragma once

#include <memory>
#include "post_process_renderer.h"

struct SSAOData
{
	glm::vec4 kernel[64];
};

class AmbientOcclusion
{
public:
	AmbientOcclusion();
	~AmbientOcclusion();
	void initialize(uint16_t width, uint16_t height);
	void shutdown();
	void profiling_gui();
	void on_window_resized(uint16_t width, uint16_t height);
	void render(uint32_t w, uint32_t h);

private:
	void render_ssao(uint32_t w, uint32_t h);
	void render_blur(uint32_t w, uint32_t h);

private:
	// Render targets
	dw::Texture2D* m_ssao_rt;
	dw::Texture2D* m_ssao_blur_rt;

	// Framebuffers
	dw::Framebuffer* m_ssao_fbo;
	dw::Framebuffer* m_ssao_blur_fbo;

	// Shaders and programs
	dw::Shader* m_ssao_vs;
	dw::Shader* m_ssao_fs;
	dw::Shader* m_ssao_blur_vs;
	dw::Shader* m_ssao_blur_fs;
	dw::Program* m_ssao_program;
	dw::Program* m_ssao_blur_program;

	std::unique_ptr<dw::Texture2D>	   m_noise_texture;
	std::unique_ptr<dw::UniformBuffer> m_kernel_ubo;

	PostProcessRenderer m_post_process_renderer;
};