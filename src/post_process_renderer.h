#pragma once

#include <ogl.h>

class PostProcessRenderer
{
public:
	PostProcessRenderer();
	~PostProcessRenderer();
	void initialize();
	void shutdown();
	void render(uint32_t w, uint32_t h, dw::Framebuffer* fbo, GLbitfield clear = GL_COLOR_BUFFER_BIT);
};