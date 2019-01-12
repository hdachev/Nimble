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

		static Framebuffer* framebuffer_for_render_targets(const uint32_t& num_render_targets, const RenderTargetView* rt_views, const RenderTargetView* depth_view);

		static void bind_render_targets(const uint32_t& num_render_targets, const RenderTargetView* rt_views, const RenderTargetView* depth_view);

		// Shader program caching.
		static std::shared_ptr<Program> create_program(const std::shared_ptr<Shader>& vs, const std::shared_ptr<Shader>& fs);
		static std::shared_ptr<Program> create_program(const std::vector<std::shared_ptr<Shader>>& shaders);

		// Uniform buffer getters.
		inline static UniformBuffer* per_view_ubo() { return m_per_view.get(); }
		inline static UniformBuffer* per_entity_ubo() { return m_per_entity.get(); }
		inline static ShaderStorageBuffer* per_scene_ssbo() { return m_per_scene.get(); }

		// Common geometry getters.
		inline static std::shared_ptr<VertexArray> cube_vao() { return m_cube_vao; }

	private:
		static void create_cube();

	private:
		// Shader and Program cache.
		static std::unordered_map<std::string, std::weak_ptr<Program>> m_program_cache;

		static StaticHashMap<uint64_t, Framebuffer*, 1024> m_fbo_cache;

		// Common geometry.
		static std::shared_ptr<VertexArray>   m_cube_vao;
		static std::shared_ptr<VertexBuffer>  m_cube_vbo;

		// Uniform buffers.
		static std::unique_ptr<UniformBuffer> m_per_view;
		static std::unique_ptr<UniformBuffer> m_per_entity;
		static std::unique_ptr<ShaderStorageBuffer> m_per_scene;
	};
}