#pragma once

#include "../render_node.h"

namespace nimble
{
class DepthOfFieldNode : public RenderNode
{
public:
    DepthOfFieldNode(RenderGraph* graph);
    ~DepthOfFieldNode();

    void        declare_connections() override;
    bool        initialize(Renderer* renderer, ResourceManager* res_mgr) override;
    void        execute(double delta, Renderer* renderer, Scene* scene, View* view) override;
    void        shutdown() override;
    std::string name() override;

private:
    void coc_generation(double delta, Renderer* renderer, Scene* scene, View* view);
    void downsample(double delta, Renderer* renderer, Scene* scene, View* view);
    void near_coc_max(double delta, Renderer* renderer, Scene* scene, View* view);
    void near_coc_blur(double delta, Renderer* renderer, Scene* scene, View* view);
    void dof_computation(double delta, Renderer* renderer, Scene* scene, View* view);
    void fill(double delta, Renderer* renderer, Scene* scene, View* view);
    void composite(double delta, Renderer* renderer, Scene* scene, View* view);

private:
    bool m_enabled = true;
    float m_near_begin  = 0.0f;
    float m_near_end    = 0.0f;
    float m_far_begin   = 200.0f;
    float m_far_end     = 250.0f;
    float m_blend       = 1.0f;
    float m_kernel_size = 1.0f;

    // Properties
    glm::vec2 m_kernel_scale;

    // CoC Pass
    std::shared_ptr<RenderTarget> m_coc_rt;

    RenderTargetView              m_coc_rtv;

    std::shared_ptr<Shader>  m_coc_fs;
    std::shared_ptr<Program> m_coc_program;

    // Downsample Pass
    std::shared_ptr<RenderTarget> m_color4_rt;
    std::shared_ptr<RenderTarget> m_mul_coc_far4_rt;
    std::shared_ptr<RenderTarget> m_coc4_rt;

    RenderTargetView              m_color4_rtv;
    RenderTargetView			  m_mul_coc_far4_rtv;
    RenderTargetView              m_coc4_rtv;

    std::shared_ptr<Shader>  m_downsample_fs;
    std::shared_ptr<Program> m_downsample_program;

    // Near CoC Max X Pass
    std::shared_ptr<RenderTarget> m_near_coc_max_x4_rt;

    RenderTargetView              m_near_coc_max_x4_rtv;

    std::shared_ptr<Shader>  m_near_coc_max_x4_fs;
    std::shared_ptr<Program> m_near_coc_max_x_program;

    // Near CoC Max Pass
    std::shared_ptr<RenderTarget> m_near_coc_max4_rt;
    RenderTargetView              m_near_coc_max4_rtv;

    std::shared_ptr<Shader>  m_near_coc_max4_fs;
    std::shared_ptr<Program> m_near_coc_max_program;

    // Near CoC Blur X Pass
    std::shared_ptr<RenderTarget> m_near_coc_blur_x4_rt;
    RenderTargetView              m_near_coc_blur_x4_rtv;

    std::shared_ptr<Shader>  m_near_coc_blur_x4_fs;
    std::shared_ptr<Program> m_near_coc_blur_x_program;

    // Near CoC Blur Pass
    std::shared_ptr<RenderTarget> m_near_coc_blur4_rt;
    RenderTargetView              m_near_coc_blur4_rtv;

    std::shared_ptr<Shader>  m_near_coc_blur4_fs;
    std::shared_ptr<Program> m_near_coc_blur_program;

    // DoF Computation Pass
    std::shared_ptr<RenderTarget> m_near_dof4_rt;
    std::shared_ptr<RenderTarget> m_far_dof4_rt;
    RenderTargetView              m_near_dof4_rtv;
    RenderTargetView              m_far_dof4_rtv;

    std::shared_ptr<Shader>  m_computation_fs;
    std::shared_ptr<Program> m_computation_program;

    // Fill Pass
    std::shared_ptr<RenderTarget> m_near_fill_dof4_rt;
    std::shared_ptr<RenderTarget> m_far_fill_dof4_rt;
    RenderTargetView              m_near_fill_dof4_rtv;
    RenderTargetView              m_far_fill_dof4_rtv;

    std::shared_ptr<Shader>  m_fill_fs;
    std::shared_ptr<Program> m_fill_program;

    // Composite Pass
    std::shared_ptr<RenderTarget> m_composite_rt;
    RenderTargetView              m_composite_rtv;

    std::shared_ptr<Shader>  m_composite_fs;
    std::shared_ptr<Program> m_composite_program;

    // Common VS
    std::shared_ptr<Shader> m_triangle_vs;

    // Inputs
    std::shared_ptr<RenderTarget> m_color_rt;
    std::shared_ptr<RenderTarget> m_depth_rt;
};

DECLARE_RENDER_NODE_FACTORY(DepthOfFieldNode);
} // namespace nimble