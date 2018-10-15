#pragma once

#include "post_process_renderer.h"
#include <vector>

class HiZBuffer
{
public:
	HiZBuffer();
	~HiZBuffer();
	void initialize(uint16_t width, uint16_t height);
	void shutdown();
	void profiling_gui();
	void on_window_resized(uint16_t width, uint16_t height);
	void render(uint32_t w, uint32_t h);

private:
	void copy_depth(uint32_t w, uint32_t h);
	void downsample(uint32_t w, uint32_t h);

private:
	bool m_enabled;
	int32_t m_first;

	dw::Texture* m_hiz_rt;
	std::vector<dw::Framebuffer*> m_fbos;

	dw::Shader*  m_quad_vs;
	dw::Shader*  m_hiz_fs;
	dw::Program* m_hiz_program;

	dw::Shader* m_copy_fs;
	dw::Program* m_copy_program;

	PostProcessRenderer m_post_process_renderer;
};