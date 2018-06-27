#pragma once

#include "post_process_renderer.h"

class Scene;

class MotionBlur
{
public:
	MotionBlur();
	~MotionBlur();
	void initialize(uint16_t width, uint16_t height);
	void shutdown();
	void on_window_resized(uint16_t width, uint16_t height);
	void render(Scene* scene, uint32_t w, uint32_t h);

private:
	PostProcessRenderer m_post_process_renderer;
	dw::Texture2D* m_motion_blur_rt;
	dw::Framebuffer* m_motion_blur_fbo;
	dw::Shader* m_motion_blur_vs;
	dw::Shader* m_motion_blur_fs;
	dw::Program* m_motion_blur_program;
};