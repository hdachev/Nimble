#pragma once

#include "../render_node.h"

namespace nimble
{
	class StandardDepthNode : public SceneRenderNode
	{
	public:
		StandardDepthNode(RenderGraph* graph);
		~StandardDepthNode();

		bool register_resources() override;
		bool initialize() override;
		void shutdown() override;
		std::string name() override;
		std::string vs_template_path() override;
		std::string fs_template_path() override;

	protected:
		void execute_internal(const View& view) override;
	};
}