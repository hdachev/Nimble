#pragma once

#include "../render_node.h"

namespace nimble
{
class AverageLuminanceNode : public RenderNode
{
public:
	AverageLuminanceNode(RenderGraph* graph);
	~AverageLuminanceNode();
	
	void        declare_connections() override;
    bool        initialize(Renderer* renderer, ResourceManager* res_mgr) override;
    void        execute(Renderer* renderer, Scene* scene, View* view) override;
    void        shutdown() override;
    std::string name() override;

private:
	std::shared_ptr<RenderTarget> m_color_rt;
    std::shared_ptr<RenderTarget> m_luminance_rt;
	RenderTargetView m_luminance_rtv;

	std::shared_ptr<Shader>  m_vs;
	std::shared_ptr<Shader>  m_fs;
	std::shared_ptr<Program> m_program;
};

DECLARE_RENDER_NODE_FACTORY(AverageLuminanceNode);
} // namespace nimble