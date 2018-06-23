#pragma once

#include <macros.h>
#include <glm.hpp>
#include <unordered_map>
#include <ogl.h>
#include <camera.h>
#include <memory>
#include "scene.h"
#include "forward_renderer.h"
#include "uniforms.h"

class Renderer
{
public:
	Renderer(uint16_t width, uint16_t height);
	~Renderer();
	void set_scene(Scene* scene);
	Scene* scene();
	void render(dw::Camera* camera);
	dw::Shader* load_shader(GLuint type, std::string& path, dw::Material* mat);
	dw::Program* load_program(std::string& combined_name, uint32_t count, dw::Shader** shaders);
	void on_window_resized(uint16_t width, uint16_t height);

	inline PerSceneUniforms* per_scene_uniform() { return &m_per_scene_uniforms; }

private:
	void update_uniforms(dw::Camera* camera);
	void create_cube();
	void create_quad();

private:
	// Current window size.
	uint16_t m_width;
	uint16_t m_height;

	// Current scene.
	Scene* m_scene;

	// Renderers
	ForwardRenderer m_forward_renderer;

	// Uniform data.
	PerFrameUniforms m_per_frame_uniforms;
	PerSceneUniforms m_per_scene_uniforms;
	PerEntityUniforms m_per_entity_uniforms[1024];
	PerMaterialUniforms m_per_material_uniforms[1024];

	// GPU Resources.
	std::unique_ptr<dw::VertexArray>   m_quad_vao;
	std::unique_ptr<dw::VertexBuffer>  m_quad_vbo;
	std::unique_ptr<dw::VertexArray>   m_cube_vao;
	std::unique_ptr<dw::VertexBuffer>  m_cube_vbo;

	dw::Shader*		   m_cube_map_vs;
	dw::Shader*		   m_cube_map_fs;
	dw::Program*       m_cube_map_program;
	dw::Shader*		   m_pssm_vs;
	dw::Shader*		   m_pssm_fs;
	dw::Program*	   m_pssm_program;
	dw::Shader*		   m_quad_vs;
	dw::Shader*		   m_quad_fs;
	dw::Program*	   m_quad_program;

	// Shader and Program cache.
	std::unordered_map<std::string, dw::Program*> m_program_cache;
	std::unordered_map<std::string, dw::Shader*> m_shader_cache;
};
