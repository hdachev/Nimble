#pragma once

#include <unordered_map>
#include <ogl.h>
#include <string>
#include <material.h>

// Class for associating names with graphics resources and offering a global point-of-access for all render passes.
class GlobalGraphicsResources
{
public:
	// Create initial resources such as the BRDF lookup table and global UBO's.
	static void initialize();

	// Cleanup all allocated resources.
	static void shutdown();

	// Lookup a previously created texture by name.
	static dw::Texture* lookup_texture(const std::string& name);

	// Texture create methods. Returns nullptr if a texture with the same name already exists,
	static dw::Texture2D* create_texture_2d(const std::string& name, const uint32_t& w, const uint32_t& h, GLenum internal_format, GLenum format, GLenum type, uint32_t num_samples = 1, uint32_t array_size = 1, uint32_t mip_levels = 1);
	static dw::TextureCube* create_texture_cube(const std::string& name, const uint32_t& w, const uint32_t& h, GLenum internal_format, GLenum format, GLenum type, uint32_t array_size = 1, uint32_t mip_levels = 1);

	// Texture destroy method.
	static void destroy_texture(const std::string& name);

	// Lookup a previously created framebuffer by name.
	static dw::Framebuffer* lookup_framebuffer(const std::string& name);

	// Framebuffer create method. Returns nullptr if a framebuffer with the same name already exists,
	static dw::Framebuffer* create_framebuffer(const std::string& name);

	// Framebuffer destroy method.
	static void destroy_framebuffer(const std::string& name);

	// Shader caching.
	static dw::Shader* load_shader(GLuint type, std::string& path, dw::Material* mat);
	static dw::Program* load_program(std::string& combined_name, uint32_t count, dw::Shader** shaders);

	// Uniform buffer getters.
	inline static dw::UniformBuffer* per_frame_ubo() { return m_per_frame; }
	inline static dw::UniformBuffer* per_scene_ubo() { return m_per_scene; }
	inline static dw::UniformBuffer* per_entity_ubo() { return m_per_entity; }

	// Common geometry getters.
	inline static dw::VertexArray* fullscreen_quad_vao() { return m_quad_vao; }
	inline static dw::VertexArray* cube_vao() { return m_cube_vao; }

private:
	static void create_cube();
	static void create_quad();

private:
	// Resource maps.
	static std::unordered_map<std::string, dw::Texture*> m_texture_map;
	static std::unordered_map<std::string, dw::Framebuffer*> m_framebuffer_map;

	// Shader and Program cache.
	static std::unordered_map<std::string, dw::Program*> m_program_cache;
	static std::unordered_map<std::string, dw::Shader*> m_shader_cache;

	// Common geometry.
	static dw::VertexArray*   m_quad_vao;
	static dw::VertexBuffer*  m_quad_vbo;
	static dw::VertexArray*   m_cube_vao;
	static dw::VertexBuffer*  m_cube_vbo;

	// Uniform buffers.
	static dw::UniformBuffer* m_per_frame;
	static dw::UniformBuffer* m_per_scene;
	static dw::UniformBuffer* m_per_entity;
};