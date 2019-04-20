#pragma once

#include "../render_node.h"

namespace nimble
{
class TAANode : public RenderNode
{
public:
    TAANode(RenderGraph* graph);
    ~TAANode();

    void        declare_connections() override;
    bool        initialize(Renderer* renderer, ResourceManager* res_mgr) override;
    void        execute(double delta, Renderer* renderer, Scene* scene, View* view) override;
    void        shutdown() override;
    std::string name() override;

private:
    bool m_enabled = true;

    // Inputs
    std::shared_ptr<RenderTarget> m_color_rt;
    std::shared_ptr<RenderTarget> m_velocity_rt;

    // Outputs
    std::shared_ptr<RenderTarget> m_taa_rt;

    RenderTargetView m_taa_rtv;

    std::shared_ptr<Shader>  m_fullscreen_triangle_vs;
    std::shared_ptr<Shader>  m_taa_fs;
    std::shared_ptr<Program> m_taa_program;
};

DECLARE_RENDER_NODE_FACTORY(TAANode);
} // namespace nimble