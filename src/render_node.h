#pragma once

#include "render_target.h"
#include "global_graphics_resources.h"

namespace nimble
{
	class RenderGraph;

	class RenderNode
	{
	public:
		RenderNode(RenderGraph* graph);

		uint32_t id();
		RenderTarget* render_target_by_name(const std::string& name);
		RenderTarget* render_target_dependecy_by_name(const std::string& name);
		Buffer* buffer_dependecy_by_name(const std::string& name);
		void set_dependency(const std::string& name, RenderTarget* rt);
		void set_dependency(const std::string& name, Buffer* buffer);

		virtual std::string name() = 0;

	protected:
		std::shared_ptr<RenderTarget> register_render_target(const std::string& name, const uint32_t& w, const uint32_t& h, GLenum target, GLenum internal_format, GLenum format, GLenum type, uint32_t num_samples = 1, uint32_t array_size = 1, uint32_t mip_levels = 1);
		std::shared_ptr<RenderTarget> register_scaled_render_target(const std::string& name, const float& w, const float& h, GLenum target, GLenum internal_format, GLenum format, GLenum type, uint32_t num_samples = 1, uint32_t array_size = 1, uint32_t mip_levels = 1);

	protected:
		RenderGraph* m_graph;
		uint32_t m_id;

	private:
		std::unordered_map<std::string, RenderTarget*> m_render_targets;
		std::unordered_map<std::string, RenderTarget*> m_rt_dependecies;
		std::unordered_map<std::string, Buffer*> m_buffer_dependecies;
	};
}