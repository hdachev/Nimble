#pragma once

#include <functional>

#include "render_target.h"
#include "global_graphics_resources.h"

namespace nimble
{
	class RenderGraph;

	class RenderNode
	{
	public:
		RenderNode(RenderGraph* graph);
		~RenderNode();

		void execute();
		void attach_sub_pass(const std::string& node_name, std::function<void(void)> function);
		RenderTarget* render_target_by_name(const std::string& name);
		RenderTarget* render_target_dependecy_by_name(const std::string& name);
		Buffer* buffer_dependecy_by_name(const std::string& name);
		void set_dependency(const std::string& name, RenderTarget* rt);
		void set_dependency(const std::string& name, Buffer* buffer);
		void timing_total(float& cpu_time, float& gpu_time);
		void timing_sub_pass(const uint32_t& index, float& cpu_time, float& gpu_time);

		// Inline getters
		inline bool is_enabled() { return m_enabled; }
		inline uint32_t id() { return m_id; }
		inline uint32_t sub_pass_count() { return m_sub_passes.size(); }

		// Inline setters
		inline void enable() { m_enabled = true; }
		inline void disable() { m_enabled = false; }

		// Virtual methods
		virtual void passthrough();
		virtual bool initialize() = 0;
		virtual void shutdown() = 0;
		virtual std::string name() = 0;

	protected:
		std::shared_ptr<RenderTarget> register_render_target(const std::string& name, const uint32_t& w, const uint32_t& h, GLenum target, GLenum internal_format, GLenum format, GLenum type, uint32_t num_samples = 1, uint32_t array_size = 1, uint32_t mip_levels = 1);
		std::shared_ptr<RenderTarget> register_scaled_render_target(const std::string& name, const float& w, const float& h, GLenum target, GLenum internal_format, GLenum format, GLenum type, uint32_t num_samples = 1, uint32_t array_size = 1, uint32_t mip_levels = 1);

	protected:
		RenderGraph* m_graph;
		
	private:
		bool m_enabled;
		uint32_t m_id;
		std::unordered_map<std::string, RenderTarget*> m_render_targets;
		std::unordered_map<std::string, RenderTarget*> m_rt_dependecies;
		std::unordered_map<std::string, Buffer*> m_buffer_dependecies;
		std::vector<std::pair<std::string, std::function<void(void)>>> m_sub_passes;
		std::vector<std::pair<float, float>> m_sub_pass_timings;
		float m_total_time_cpu;
		float m_total_time_gpu;
		std::string m_passthrough_name;
	};
}