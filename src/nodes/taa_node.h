#pragma once

#include "../render_node.h"

namespace nimble
{
class TAANode : public RenderNode
{
public:
    enum Neighborhood
    {
        MIN_MAX_3X3,
        MIN_MAX_3X3_ROUNDED,
        MIN_MAX_4_TAP_VARYING
    };

    TAANode(RenderGraph* graph);
    ~TAANode();

    void        declare_connections() override;
    bool        initialize(Renderer* renderer, ResourceManager* res_mgr) override;
    void        execute(double delta, Renderer* renderer, Scene* scene, View* view) override;
    void        shutdown() override;
    std::string name() override;

private:
    bool m_enabled = true;
    Neighborhood m_neighborhood = MIN_MAX_3X3_ROUNDED;
    bool m_unjitter_color_samples = true;
    bool m_unjitter_neighborhood = false;
    bool m_unjitter_reprojection = false;
    bool m_use_ycocg= false;
    bool m_use_clipping = true;
    bool m_use_dilation = true;
    bool m_use_motion_blur = true;
    bool m_use_optimizations = true;
    float        m_feedback_min          = 0.88f;
    float        m_feedback_max          = 0.97f;

    // Inputs
    std::shared_ptr<RenderTarget> m_color_rt;
    std::shared_ptr<RenderTarget> m_prev_rt;
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