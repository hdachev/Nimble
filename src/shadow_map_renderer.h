#pragma once

#include <camera.h>
#include "scene_renderer.h"

class Scene;
class CSM;

class ShadowMapRenderer
{
public:
	ShadowMapRenderer();
	~ShadowMapRenderer();
	void render(Scene* scene, CSM* csm_technique);

private:
	SceneRenderer m_scene_renderer;
	dw::Shader* m_csm_vs;
	dw::Shader* m_csm_fs;
	dw::Program* m_csm_program;
};