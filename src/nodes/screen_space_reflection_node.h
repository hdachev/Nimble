#pragma once

#include "../render_node.h"

namespace nimble
{
class ScreenSpaceReflectionNode : public RenderNode
{
public:
    ScreenSpaceReflectionNode(RenderGraph* graph);
    ~ScreenSpaceReflectionNode();

    void        declare_connections() override;
    bool        initialize(Renderer* renderer, ResourceManager* res_mgr) override;
    void        execute(double delta, Renderer* renderer, Scene* scene, View* view) override;
    void        shutdown() override;
    std::string name() override;

private:
    bool  m_enabled;

	// Inputs
    std::shared_ptr<RenderTarget> m_hiz_depth_rt;
    std::shared_ptr<RenderTarget> m_metallic_rt;

	// Outputs
    std::shared_ptr<RenderTarget> m_ssr_rt;

    RenderTargetView m_ssr_rtv;

    std::shared_ptr<Shader>  m_ssr_cs;
    std::shared_ptr<Program> m_ssr_program;
};

DECLARE_RENDER_NODE_FACTORY(ScreenSpaceReflectionNode);
} // namespace nimble