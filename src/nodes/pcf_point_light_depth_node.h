#pragma once

#include "../render_node.h"

namespace nimble
{
	class PCFPointLightDepthNode : public SceneRenderNode
	{
	public:
		PCFPointLightDepthNode(RenderGraph* graph);
		~PCFPointLightDepthNode();

		bool register_resources() override;
		bool initialize() override;
		void shutdown() override;
		void set_shader_uniforms(const View* view, Program* program, int32_t& tex_unit) override;
		std::string name() override;
		std::string vs_template_path() override;
		std::string fs_template_path() override;

	protected:
		void execute_internal(const View* view) override;
	};
}