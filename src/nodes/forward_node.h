#pragma once

#include "../render_node.h"

namespace nimble
{
class ForwardNode : public SceneRenderNode
{
public:
    ForwardNode(RenderGraph* graph);
    ~ForwardNode();

    bool        register_resources() override;
    bool        initialize() override;
    void        shutdown() override;
    std::string name() override;
    std::string vs_template_path() override;
    std::string fs_template_path() override;

protected:
    void execute_internal(const View* view) override;

private:
    std::shared_ptr<RenderTarget> m_color_rt;
    std::shared_ptr<RenderTarget> m_depth_rt;
    std::shared_ptr<RenderTarget> m_velocity_rt;
    RenderTargetView              m_color_rtv;
    RenderTargetView              m_velocity_rtv;
    RenderTargetView              m_depth_rtv;
};

DECLARE_RENDER_NODE_FACTORY(ForwardNode);
} // namespace nimble