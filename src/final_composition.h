#pragma once

#include "camera.h"
#include "post_process_renderer.h"

namespace nimble
{
	class FinalComposition
	{
	public:
		FinalComposition();
		~FinalComposition();
		void render(Camera* camera, uint32_t w, uint32_t h);

	private:
		Shader*		    m_composition_vs;
		Shader*		    m_composition_fs;
		Program*	    m_composition_program;
		PostProcessRenderer m_post_process_renderer;
	};
}
