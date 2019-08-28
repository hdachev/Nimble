#pragma once

#include "../render_node.h"

namespace nimble
{
class FXAANode : public RenderNode
{
public:
    FXAANode(RenderGraph* graph);
    ~FXAANode();

    void        declare_connections() override;
    bool        initialize_private(Renderer* renderer, ResourceManager* res_mgr) override;
    void        execute(double delta, Renderer* renderer, Scene* scene, View* view) override;
    void        shutdown() override;
    std::string name() override;

private:
    bool  m_enabled                    = true;
    float m_quality_edge_threshold     = 0.166f;
    float m_quality_edge_threshold_min = 0.0833f;

    // Inputs
    std::shared_ptr<RenderTarget> m_color_rt;

    // Outputs
    std::shared_ptr<RenderTarget> m_fxaa_rt;

    RenderTargetView m_fxaa_rtv;

    std::shared_ptr<Shader>  m_fullscreen_triangle_vs;
    std::shared_ptr<Shader>  m_fxaa_fs;
    std::shared_ptr<Program> m_fxaa_program;
};

DECLARE_RENDER_NODE_FACTORY(FXAANode);
} // namespace nimble