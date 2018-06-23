#include "post_process_renderer.h"
#include "global_graphics_resources.h"

// -----------------------------------------------------------------------------------------------------------------------------------

PostProcessRenderer::PostProcessRenderer() {}

// -----------------------------------------------------------------------------------------------------------------------------------

PostProcessRenderer::~PostProcessRenderer() {}

// -----------------------------------------------------------------------------------------------------------------------------------

void PostProcessRenderer::render(uint32_t w, uint32_t h, dw::Framebuffer* fbo)
{
	// Set state.
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	glCullFace(GL_NONE);

	// Bind framebuffer.
	if (fbo)
		fbo->bind();
	else
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// Set and clear viewport.
	glViewport(0, 0, w, h);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	// Bind VAO.
	GlobalGraphicsResources::fullscreen_quad_vao()->bind();

	// Draw quad.
	glDrawArrays(GL_TRIANGLES, 0, 6);
}

// -----------------------------------------------------------------------------------------------------------------------------------