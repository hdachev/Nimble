#pragma once

#include "../render_node.h"

namespace nimble
{
class ForwardNode : public RenderNode
{
public:
    ForwardNode(RenderGraph* graph);
    ~ForwardNode();

    void        declare_connections() override;
    bool        initialize(Renderer* renderer, ResourceManager* res_mgr) override;
    void        execute(double delta, Renderer* renderer, Scene* scene, View* view) override;
    void        shutdown() override;
    std::string name() override;

private:
    std::shared_ptr<ShaderLibrary> m_library;
    std::shared_ptr<RenderTarget>  m_color_rt;
    std::shared_ptr<RenderTarget>  m_depth_rt;
    std::shared_ptr<RenderTarget>  m_velocity_rt;
    RenderTargetView               m_color_rtv[2];
    RenderTargetView               m_depth_rtv;
};

DECLARE_RENDER_NODE_FACTORY(ForwardNode);
} // namespace nimble