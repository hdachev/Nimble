#pragma once

#include <unordered_map>
#include <string>
#include "ogl.h"
#include "material.h"
#include "uniforms.h"
#include "render_target.h"

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

		// Render Target creation methods. Actual texture is created during initialize_render_targets().
		static std::shared_ptr<RenderTarget> request_render_target(const uint32_t& graph_id, const uint32_t& node_id, const uint32_t& w, const uint32_t& h, GLenum target, GLenum internal_format, GLenum format, GLenum type, uint32_t num_samples = 1, uint32_t array_size = 1, uint32_t mip_levels = 1);

		// Scaled variant. Uses a normalized float value to represent the w/h ratio to the w/h of the window.
		static std::shared_ptr<RenderTarget> request_scaled_render_target(const uint32_t& graph_id, const uint32_t& node_id, const float& w, const float& h, GLenum target, GLenum internal_format, GLenum format, GLenum type, uint32_t num_samples = 1, uint32_t array_size = 1, uint32_t mip_levels = 1);

		static void initialize_render_targets();

		// Shader caching.
		static std::shared_ptr<Program> load_program(const std::shared_ptr<Shader>& vs, const std::shared_ptr<Shader>& fs);
		static std::shared_ptr<Program> load_program(const std::vector<std::shared_ptr<Shader>>& shaders);

		// Uniform buffer getters.
		inline static UniformBuffer* per_frame_ubo() { return m_per_frame; }
		inline static UniformBuffer* per_scene_ubo() { return m_per_scene; }
		inline static UniformBuffer* per_entity_ubo() { return m_per_entity; }

		// Common geometry getters.
		inline static VertexArray* fullscreen_quad_vao() { return m_quad_vao; }
		inline static VertexArray* cube_vao() { return m_cube_vao; }

		// Uniform getters.
		inline static PerFrameUniforms& per_frame_uniforms() { return m_per_frame_uniforms; }

	private:
		static void create_cube();
		static void create_quad();
		static bool read_shader(std::string path, std::string& out, const std::vector<std::string> defines = std::vector<std::string>());

	private:
		// Resource maps.
		static std::vector<std::weak_ptr<RenderTarget>> m_render_target_pool;

		// Shader and Program cache.
		static std::unordered_map<std::string, std::weak_ptr<Program>> m_program_cache;

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