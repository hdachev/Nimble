#pragma once

#include "render_node.h"
#include "view.h"

namespace nimble
{
	class Renderer;

	enum RenderGraphType : uint32_t
	{
		RENDER_GRAPH_STANDARD,
		RENDER_GRAPH_SHADOW
	};

	class RenderGraph
	{
	public:
		RenderGraph(Renderer* renderer);
		~RenderGraph();

		void build(std::shared_ptr<RenderNode> end_node);
		void execute(const View* view);
		void shutdown();
		void clear();
		std::shared_ptr<RenderNode> node_by_name(const std::string& name);
		void on_window_resized(const uint32_t& w, const uint32_t& h);

		inline void set_name(const std::string& name) { m_name = name; }
		inline std::string name() { return m_name; }
		inline uint32_t node_count() { return m_flattened_graph.size(); }
		inline std::shared_ptr<RenderNode> node(const uint32_t& idx) { return m_flattened_graph[idx]; }
		inline uint32_t window_width() { return m_window_width; }
		inline uint32_t window_height() { return m_window_height; }
		inline Renderer* renderer() { return m_renderer; }
		inline virtual uint32_t actual_viewport_width() { return window_width(); }
		inline virtual uint32_t actual_viewport_height() { return window_height(); }
		inline virtual uint32_t rendered_viewport_width() { return actual_viewport_width(); }
		inline virtual uint32_t rendered_viewport_height() { return actual_viewport_height(); }

		virtual bool initialize();
		virtual RenderGraphType type();

	private:
		void flatten_graph();
		void traverse_and_push_node(std::shared_ptr<RenderNode> node);
		bool is_node_pushed(std::shared_ptr<RenderNode> node);

	private:
		std::string m_name;
		uint32_t m_window_width;
		uint32_t m_window_height;
		std::shared_ptr<RenderNode> m_end_node;
		std::vector<std::shared_ptr<RenderNode>> m_flattened_graph;
		Renderer* m_renderer;
	};

	class ShadowRenderGraph : public RenderGraph
	{
	public:
		ShadowRenderGraph(Renderer* renderer);

		bool initialize() override;

		RenderGraphType type() override;

		inline std::string sampling_source() { return m_sampling_source; }

		virtual std::string sampling_source_path() = 0;

	private:
		std::string m_sampling_source;
	};
}