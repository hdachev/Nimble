#pragma once

#include "post_process_renderer.h"

class Bloom
{
public:
	Bloom();
	~Bloom();
	void initialize(uint16_t width, uint16_t height);
	void shutdown();
	void profiling_gui();
	void on_window_resized(uint16_t width, uint16_t height);
	void render(uint32_t w, uint32_t h);

private:
	void downsample(uint32_t w, uint32_t h);
	void blur(uint32_t w, uint32_t h);
	void composite(uint32_t w, uint32_t h);
	void separable_blur(dw::Texture* src_rt, dw::Framebuffer* dest_fbo, dw::Texture* temp_rt, dw::Framebuffer* temp_fbo, uint32_t w, uint32_t h);

private:
	dw::Texture* m_bloom_rt[4][2]; 
	dw::Framebuffer* m_bloom_fbo[4][2]; 

	// Downsample shader
	dw::Shader*  m_bloom_downsample_vs;
	dw::Shader*  m_bloom_downsample_fs;
	dw::Program* m_bloom_downsample_program;

	// Blur shader
	dw::Shader*  m_bloom_blur_vs;
	dw::Shader*  m_bloom_blur_fs;
	dw::Program* m_bloom_blur_program;

	// Composite shader
	dw::Shader*  m_bloom_composite_vs;
	dw::Shader*  m_bloom_composite_fs;
	dw::Program* m_bloom_composite_program;

	PostProcessRenderer m_post_process_renderer;
};