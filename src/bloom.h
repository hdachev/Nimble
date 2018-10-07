#pragma once

#include "post_process_renderer.h"

#define BLOOM_TEX_CHAIN_SIZE 5

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

	inline float threshold() { return m_threshold; }
	inline void set_threshold(float t) { m_threshold = t; }

	inline float strength() { return m_strength; }
	inline void set_strength(float s) { m_strength = s; }

	inline bool is_enabled() { return m_enabled; }
	inline void enable() { m_enabled = true; }
	inline void disable() { m_enabled = false; }

private:
	void bright_pass(uint32_t w, uint32_t h);
	void downsample(uint32_t w, uint32_t h);
	void upsample(uint32_t w, uint32_t h);
	void composite(uint32_t w, uint32_t h);

private:
	float m_threshold;
	float m_strength;
	bool m_enabled;

	dw::Texture* m_composite_rt;
	dw::Framebuffer* m_composite_fbo;
	dw::Texture* m_bloom_rt[BLOOM_TEX_CHAIN_SIZE]; 
	dw::Framebuffer* m_bloom_fbo[BLOOM_TEX_CHAIN_SIZE]; 

	dw::Shader*  m_quad_vs;

	// Brightpass shader
	dw::Shader*  m_bright_pass_fs;
	dw::Program* m_bright_pass_program;

	// Downsample shader
	dw::Shader*  m_bloom_downsample_fs;
	dw::Program* m_bloom_downsample_program;

	// Upsample shader
	dw::Shader*  m_bloom_upsample_fs;
	dw::Program* m_bloom_upsample_program;

	// Composite shader
	dw::Shader*  m_bloom_composite_fs;
	dw::Program* m_bloom_composite_program;

	PostProcessRenderer m_post_process_renderer;
};