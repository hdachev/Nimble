#pragma once

#include "camera.h"
#include "scene_renderer.h"

namespace nimble
{
	class Scene;
	class CSM;

	class ShadowMapRenderer
	{
	public:
		ShadowMapRenderer();
		~ShadowMapRenderer();
		void profiling_gui();
		void render(Scene* scene, CSM* csm_technique);

	private:
		SceneRenderer m_scene_renderer;
		Shader* m_csm_vs;
		Shader* m_csm_fs;
		Program* m_csm_program;
	};
}