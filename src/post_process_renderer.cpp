#include "post_process_renderer.h"
#include "global_graphics_resources.h"

namespace nimble
{
	// -----------------------------------------------------------------------------------------------------------------------------------

	PostProcessRenderer::PostProcessRenderer() {}

	// -----------------------------------------------------------------------------------------------------------------------------------

	PostProcessRenderer::~PostProcessRenderer() {}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void PostProcessRenderer::render(uint32_t w, uint32_t h, Framebuffer* fbo, GLbitfield clear)
	{
		// Set state.
		if (clear & GL_DEPTH_BUFFER_BIT)
			glEnable(GL_DEPTH_TEST);
		else
			glDisable(GL_DEPTH_TEST);
		
		glDisable(GL_CULL_FACE);

		// Bind framebuffer.
		if (fbo)
			fbo->bind();
		else
			glBindFramebuffer(GL_FRAMEBUFFER, 0);

		// Set and clear viewport.
		glViewport(0, 0, w, h);

		if (clear != 0)
		{
			glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
			glClear(clear);
		}
		
		// Bind VAO.
		GlobalGraphicsResources::fullscreen_quad_vao()->bind();

		// Draw quad.
		glDrawArrays(GL_TRIANGLES, 0, 6);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------
}