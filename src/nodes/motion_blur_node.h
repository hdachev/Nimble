#pragma once

#include "../render_node.h"

namespace nimble
{
class MotionBlurNode : public RenderNode
{
public:
    MotionBlurNode(RenderGraph* graph);
    ~MotionBlurNode();

    void        declare_connections() override;
    bool        initialize(Renderer* renderer, ResourceManager* res_mgr) override;
    void        execute(double delta, Renderer* renderer, Scene* scene, View* view) override;
    void        shutdown() override;
    std::string name() override;

private:
    std::shared_ptr<RenderTarget> m_color_rt;
    std::shared_ptr<RenderTarget> m_velocity_rt;

    std::shared_ptr<RenderTarget> m_motion_blur_rt;
    RenderTargetView              m_motion_blur_rtv;

    std::shared_ptr<Shader>  m_vs;
    std::shared_ptr<Shader>  m_fs;
    std::shared_ptr<Program> m_program;

    bool    m_enabled     = true;
    int32_t m_num_samples = 32;
};

DECLARE_RENDER_NODE_FACTORY(MotionBlurNode);
} // namespace nimble