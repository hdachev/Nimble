#pragma once

#include <unordered_map>
#include <string>
#include "ogl.h"
#include "material.h"
#include "uniforms.h"

namespace nimble
{
	enum RendererType
	{
		RENDERER_FORWARD = 0,
		RENDERER_DEFERRED = 1
	};

	// Class for associating names with graphics resources and offering a global point-of-access for all render passes.
	class GlobalGraphicsResources
	{
	public:
		// Create initial resources such as the BRDF lookup table and global UBO's.
		static void initialize();

		// Cleanup all allocated resources.
		static void shutdown();

		// Lookup a previously created texture by name.
		static std::shared_ptr<Texture> lookup_texture(const std::string& name);

		// Texture create methods. Returns nullptr if a texture with the same name already exists,
		static std::shared_ptr<Texture2D> create_texture_2d(const std::string& name, const uint32_t& w, const uint32_t& h, GLenum internal_format, GLenum format, GLenum type, uint32_t num_samples = 1, uint32_t array_size = 1, uint32_t mip_levels = 1);
		static std::shared_ptr<TextureCube> create_texture_cube(const std::string& name, const uint32_t& w, const uint32_t& h, GLenum internal_format, GLenum format, GLenum type, uint32_t array_size = 1, uint32_t mip_levels = 1);

		// Texture destroy method.
		static void destroy_texture(const std::string& name);

		// Lookup a previously created framebuffer by name.
		static std::shared_ptr<Framebuffer> lookup_framebuffer(const std::string& name);

		// Framebuffer create method. Returns nullptr if a framebuffer with the same name already exists,
		static std::shared_ptr<Framebuffer> create_framebuffer(const std::string& name);

		// Framebuffer destroy method.
		static void destroy_framebuffer(const std::string& name);

		// Shader caching.
		static Program* load_program(std::string& combined_name, uint32_t count, Shader** shaders);

		// Uniform buffer getters.
		inline static UniformBuffer* per_frame_ubo() { return m_per_frame; }
		inline static UniformBuffer* per_scene_ubo() { return m_per_scene; }
		inline static UniformBuffer* per_entity_ubo() { return m_per_entity; }

		// Common geometry getters.
		inline static VertexArray* fullscreen_quad_vao() { return m_quad_vao; }
		inline static VertexArray* cube_vao() { return m_cube_vao; }

		// Uniform getters.
		inline static PerFrameUniforms& per_frame_uniforms() { return m_per_frame_uniforms; }

		static std::string combined_program_name(const std::string& vs, const std::string& fs, std::vector<std::string> defines);

	private:
		static void create_cube();
		static void create_quad();
		static bool read_shader(std::string path, std::string& out, const std::vector<std::string> defines = std::vector<std::string>());

	private:
		// Resource maps.
		static std::unordered_map<std::string, Texture*> m_texture_map;
		static std::unordered_map<std::string, Framebuffer*> m_framebuffer_map;

		// Shader and Program cache.
		static std::unordered_map<std::string, Program*> m_program_cache;
		static std::unordered_map<std::string, Shader*> m_shader_cache;

		// Common geometry.
		static VertexArray*   m_quad_vao;
		static VertexBuffer*  m_quad_vbo;
		static VertexArray*   m_cube_vao;
		static VertexBuffer*  m_cube_vbo;

		// Uniform buffers.
		static UniformBuffer* m_per_frame;
		static UniformBuffer* m_per_scene;
		static UniformBuffer* m_per_entity;

		// Per Frame uniforms.
		static PerFrameUniforms m_per_frame_uniforms;
	};
}