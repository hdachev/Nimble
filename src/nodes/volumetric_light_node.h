#pragma once

#include "../render_node.h"

namespace nimble
{
class VolumetricLightNode : public RenderNode
{
public:
    VolumetricLightNode(RenderGraph* graph);
    ~VolumetricLightNode();

    void        declare_connections() override;
    bool        initialize(Renderer* renderer, ResourceManager* res_mgr) override;
    void        execute(double delta, Renderer* renderer, Scene* scene, View* view) override;
    void        shutdown() override;
    std::string name() override;

private:
    void volumetrics(Renderer* renderer, Scene* scene, View* view);

private:
    std::shared_ptr<RenderTarget> m_color_rt;
    std::shared_ptr<RenderTarget> m_depth_rt;

    std::shared_ptr<RenderTarget> m_volumetrics_rt;
    RenderTargetView              m_volumetrics_rtv;

    std::shared_ptr<Shader>  m_volumetrics_vs;
    std::shared_ptr<Shader>  m_volumetrics_fs;
    std::shared_ptr<Program> m_volumetrics_program;

    int32_t m_num_samples = 32;
    float   m_mie_g       = 0.1f;
};

DECLARE_RENDER_NODE_FACTORY(MotionBlurNode);
} // namespace nimble