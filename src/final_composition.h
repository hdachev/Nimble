#pragma once

#include <camera.h>
#include "post_process_renderer.h"

class FinalComposition
{
public:
	FinalComposition();
	~FinalComposition();
	void render(dw::Camera* camera, uint32_t w, uint32_t h, int current_output = 0);

private:
	dw::Shader*		    m_composition_vs;
	dw::Shader*		    m_composition_fs;
	dw::Program*	    m_composition_program;
	PostProcessRenderer m_post_process_renderer;
};