#pragma once

#include "render_node.h"

namespace nimble
{
	class RenderGraph
	{
	public:
		RenderGraph();
		~RenderGraph();

		bool initialize();
		void shutdown();
		void clear();
		void execute();
		uint32_t id();
		bool attach_and_initialize_node(std::shared_ptr<RenderNode> node);
		std::shared_ptr<RenderNode> node_by_name(const std::string& name);

		virtual std::string name() = 0;
		virtual bool build() = 0;
		virtual void refresh() = 0;

	private:
		uint32_t m_id;
		std::vector<std::shared_ptr<RenderNode>> m_nodes;
	};
}