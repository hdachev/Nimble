#pragma once

#include <unordered_map>
#include <string>
#include "ogl.h"
#include "material.h"
#include "uniforms.h"
#include "render_target.h"
#include "static_hash_map.h"

namespace nimble
{
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

		// Actually creates the texture associated with the render target. Used during initialization and window resizes.
		static void initialize_render_targets(const uint32_t& window_w, const uint32_t& window_h);

		static Framebuffer* framebuffer_for_render_targets(const uint32_t& num_render_targets, const RenderTargetView* rt_views, const RenderTargetView* depth_view);

		static void bind_render_targets(const uint32_t& num_render_targets, const RenderTargetView* rt_views, const RenderTargetView* depth_view);

		// Shader program caching.
		static std::shared_ptr<Program> create_program(const std::shared_ptr<Shader>& vs, const std::shared_ptr<Shader>& fs);
		static std::shared_ptr<Program> create_program(const std::vector<std::shared_ptr<Shader>>& shaders);

		// Uniform buffer getters.
		inline static UniformBuffer* per_view_ubo() { return m_per_view.get(); }
		inline static UniformBuffer* per_scene_ubo() { return m_per_scene.get(); }
		inline static UniformBuffer* per_entity_ubo() { return m_per_entity.get(); }

		// Common geometry getters.
		inline static std::shared_ptr<VertexArray> cube_vao() { return m_cube_vao; }

	private:
		static void create_cube();

	private:
		// Resource maps.
		static std::vector<std::weak_ptr<RenderTarget>> m_render_target_pool;

		// Shader and Program cache.
		static std::unordered_map<std::string, std::weak_ptr<Program>> m_program_cache;

		static StaticHashMap<uint64_t, Framebuffer*, 1024> m_fbo_cache;

		// Common geometry.
		static std::shared_ptr<VertexArray>   m_cube_vao;
		static std::shared_ptr<VertexBuffer>  m_cube_vbo;

		// Uniform buffers.
		static std::unique_ptr<UniformBuffer> m_per_view;
		static std::unique_ptr<UniformBuffer> m_per_scene;
		static std::unique_ptr<UniformBuffer> m_per_entity;
	};
}