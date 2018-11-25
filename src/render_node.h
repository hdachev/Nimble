#pragma once

#include <functional>

#include "render_target.h"
#include "global_graphics_resources.h"

namespace nimble
{
	class RenderGraph;

	enum RenderNodeType
	{
		RENDER_NODE_GEOMETRY = 0,
		RENDER_NODE_FULLSCREEN = 1,
		RENDER_NODE_COMPUTE = 2
	};

	class RenderNode
	{
	public:
		RenderNode(RenderNodeType type, RenderGraph* graph);
		~RenderNode();

		RenderTarget* render_target_by_name(const std::string& name);
		RenderTarget* render_target_dependecy_by_name(const std::string& name);
		Buffer* buffer_dependecy_by_name(const std::string& name);
		void set_dependency(const std::string& name, RenderTarget* rt);
		void set_dependency(const std::string& name, Buffer* buffer);
		void timing_total(float& cpu_time, float& gpu_time);

		// Inline getters
		inline RenderNodeType type() { return m_render_node_type; }
		inline bool is_enabled() { return m_enabled; }
		inline uint32_t id() { return m_id; }

		// Inline setters
		inline void enable() { m_enabled = true; }
		inline void disable() { m_enabled = false; }

		// Virtual methods
		virtual void passthrough();
		virtual void execute() = 0;
		virtual bool initialize() = 0;
		virtual void shutdown() = 0;
		virtual std::string name() = 0;

	protected:
		std::shared_ptr<RenderTarget> register_render_target(const std::string& name, const uint32_t& w, const uint32_t& h, GLenum target, GLenum internal_format, GLenum format, GLenum type, uint32_t num_samples = 1, uint32_t array_size = 1, uint32_t mip_levels = 1);
		std::shared_ptr<RenderTarget> register_scaled_render_target(const std::string& name, const float& w, const float& h, GLenum target, GLenum internal_format, GLenum format, GLenum type, uint32_t num_samples = 1, uint32_t array_size = 1, uint32_t mip_levels = 1);

	protected:
		RenderGraph* m_graph;
		float m_total_time_cpu;
		float m_total_time_gpu;
		std::string m_passthrough_name;
		
	private:
		bool m_enabled;
		uint32_t m_id;
		RenderNodeType m_render_node_type;
		std::unordered_map<std::string, RenderTarget*> m_render_targets;
		std::unordered_map<std::string, RenderTarget*> m_rt_dependecies;
		std::unordered_map<std::string, Buffer*> m_buffer_dependecies;
	};

	class GeometryRenderNode : public RenderNode
	{
	public:
		GeometryRenderNode(RenderGraph* graph);
		~GeometryRenderNode();

		void execute() override;
	};

	class FullscreenRenderNode : public RenderNode
	{
	public:
		FullscreenRenderNode(RenderGraph* graph);
		~FullscreenRenderNode();

		void execute() override;
		void attach_sub_pass(const std::string& node_name, std::function<void(void)> function);
		void timing_sub_pass(const uint32_t& index, float& cpu_time, float& gpu_time);

		// Inline getters
		inline uint32_t sub_pass_count() { return m_sub_passes.size(); }

	private:
		std::vector<std::pair<std::string, std::function<void(void)>>> m_sub_passes;
		std::vector<std::pair<float, float>> m_sub_pass_timings;
	};
}