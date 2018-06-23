#pragma once

#include <macros.h>
#include <glm.hpp>
#include <unordered_map>
#include <ogl.h>
#include <camera.h>
#include <memory>
#include "scene.h"
#include "forward_renderer.h"
#include "final_composition.h"
#include "uniforms.h"

class Renderer
{
public:
	Renderer(uint16_t width, uint16_t height);
	~Renderer();
	void set_scene(Scene* scene);
	Scene* scene();
	void render(dw::Camera* camera);
	void on_window_resized(uint16_t width, uint16_t height);

	inline PerSceneUniforms* per_scene_uniform() { return &m_per_scene_uniforms; }

private:
	void update_uniforms(dw::Camera* camera);

private:
	// Current window size.
	uint16_t m_width;
	uint16_t m_height;

	// Current scene.
	Scene* m_scene;

	// Renderers
	ForwardRenderer m_forward_renderer;
	FinalComposition m_final_composition;

	// Uniform data.
	PerFrameUniforms m_per_frame_uniforms;
	PerSceneUniforms m_per_scene_uniforms;
	PerEntityUniforms m_per_entity_uniforms[1024];
	PerMaterialUniforms m_per_material_uniforms[1024];
};
