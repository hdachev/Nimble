#pragma once

#include "../render_node.h"

namespace nimble
{
class DepthPrepassNode : public RenderNode
{
public:
    DepthPrepassNode(RenderGraph* graph);
    ~DepthPrepassNode();

    void        declare_connections() override;
    bool        initialize(Renderer* renderer, ResourceManager* res_mgr) override;
    void        execute(double delta, Renderer* renderer, Scene* scene, View* view) override;
    void        shutdown() override;
    std::string name() override;

private:
    std::shared_ptr<GeneratedShaderLibrary> m_library;
    std::shared_ptr<RenderTarget>           m_depth_rt;
    RenderTargetView                        m_depth_rtv;
};

DECLARE_RENDER_NODE_FACTORY(DepthPrepassNode);
} // namespace nimble