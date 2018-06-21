#pragma once

#include <macros.h>
#include <glm.hpp>
#include <unordered_map>
#include <ogl.h>
#include <camera.h>
#include <memory>
#include "scene.h"

#define MAX_POINT_LIGHTS 32
#define MAX_SHADOW_FRUSTUM 8

struct PointLight
{
	DW_ALIGNED(16) glm::vec4 position;
	DW_ALIGNED(16) glm::vec4 color;
};

struct DirectionalLight
{
	DW_ALIGNED(16) glm::vec4 direction;
	DW_ALIGNED(16) glm::vec4 color;
};

struct ShadowFrustum
{
	DW_ALIGNED(16) glm::mat4 shadowMatrix;
	DW_ALIGNED(16) float	 farPlane;
};

struct PerFrameUniforms
{
	DW_ALIGNED(16) glm::mat4	 lastViewProj;
	DW_ALIGNED(16) glm::mat4	 viewProj;
	DW_ALIGNED(16) glm::mat4	 invViewProj;
	DW_ALIGNED(16) glm::mat4	 projMat;
	DW_ALIGNED(16) glm::mat4	 viewMat;
	DW_ALIGNED(16) glm::vec4	 viewPos;
	DW_ALIGNED(16) glm::vec4	 viewDir;
	DW_ALIGNED(16) int			 numCascades;
	DW_ALIGNED(16) ShadowFrustum shadowFrustums[MAX_SHADOW_FRUSTUM];
};

struct PerEntityUniforms
{
	DW_ALIGNED(16) glm::mat4 mvpMat;
	DW_ALIGNED(16) glm::mat4 lastMvpMat;
	DW_ALIGNED(16) glm::mat4 modalMat;
	DW_ALIGNED(16) glm::vec4 worldPos;
	uint8_t	  padding[48];
};

struct PerSceneUniforms
{
	DW_ALIGNED(16) PointLight 		pointLights[MAX_POINT_LIGHTS];
	DW_ALIGNED(16) DirectionalLight directionalLight;
	DW_ALIGNED(16) int				pointLightCount;
};

struct PerMaterialUniforms
{
	DW_ALIGNED(16) glm::vec4 albedoValue;
	DW_ALIGNED(16) glm::vec4 metalnessRoughness;
};

struct PerFrustumSplitUniforms
{
	DW_ALIGNED(16) glm::mat4 crop_matrix;
};

class Renderer
{
public:
	Renderer(uint16_t width, uint16_t height);
	~Renderer();
	void set_scene(Scene* scene);
	Scene* scene();
	void render(dw::Camera* camera, uint16_t w = 0, uint16_t h = 0, dw::Framebuffer* fbo = nullptr);
	dw::Shader* load_shader(GLuint type, std::string& path, dw::Material* mat);
	dw::Program* load_program(std::string& combined_name, uint32_t count, dw::Shader** shaders);

	inline PerSceneUniforms* per_scene_uniform() { return &m_per_scene_uniforms; }

private:
	void create_cube();
	void create_quad();
	void render_scene(uint16_t w = 0, uint16_t h = 0, dw::Framebuffer* fbo = nullptr);

private:
	// Current window size.
	uint16_t m_width;
	uint16_t m_height;

	// Current scene.
	Scene* m_scene;

	// Uniform data.
	PerFrameUniforms m_per_frame_uniforms;
	PerSceneUniforms m_per_scene_uniforms;
	PerEntityUniforms m_per_entity_uniforms[1024];
	PerMaterialUniforms m_per_material_uniforms[1024];

	// GPU Resources.
	std::unique_ptr<dw::UniformBuffer> m_per_scene;
	std::unique_ptr<dw::UniformBuffer> m_per_frame;
	std::unique_ptr<dw::UniformBuffer> m_per_material;
	std::unique_ptr<dw::UniformBuffer> m_per_entity;
	std::unique_ptr<dw::UniformBuffer> m_per_frustum_split;
	std::unique_ptr<dw::VertexArray>   m_quad_vao;
	std::unique_ptr<dw::VertexBuffer>  m_quad_vbo;
	std::unique_ptr<dw::VertexArray>   m_cube_vao;
	std::unique_ptr<dw::VertexBuffer>  m_cube_vbo;
	std::unique_ptr<dw::Texture2D>	   m_brdfLUT;
	std::unique_ptr<dw::Texture2D>	   m_color_buffer;
	std::unique_ptr<dw::Texture2D>	   m_depth_buffer;
	std::unique_ptr<dw::Framebuffer>   m_color_fbo;

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
