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
    void blur(Renderer* renderer, Scene* scene, View* view);
    void upscale(Renderer* renderer, Scene* scene, View* view);

private:
    std::shared_ptr<RenderTarget> m_color_rt;
    std::shared_ptr<RenderTarget> m_depth_rt;

    std::shared_ptr<RenderTarget> m_volumetric_light_rt;
    std::shared_ptr<RenderTarget> m_h_blur_rt;
    std::shared_ptr<RenderTarget> m_v_blur_rt;

    RenderTargetView m_volumetrics_rtv;
    RenderTargetView m_h_blur_rtv;
    RenderTargetView m_v_blur_rtv;
    RenderTargetView m_upscale_rtv;

    std::shared_ptr<Shader> m_fullscreen_triangle_vs;

    std::shared_ptr<Shader>  m_volumetrics_fs;
    std::shared_ptr<Program> m_volumetrics_program;

    std::shared_ptr<Shader>  m_blur_fs;
    std::shared_ptr<Program> m_blur_program;

    std::shared_ptr<Shader>  m_upscale_fs;
    std::shared_ptr<Program> m_upscale_program;

    std::unique_ptr<Texture2D> m_dither_texture;

    uint32_t m_flags       = 0;
    bool     m_enabled     = true;
    bool     m_dither      = true;
    bool     m_blur        = true;
    int32_t  m_num_samples = 32;
    float    m_mie_g       = 0.1f;
};

DECLARE_RENDER_NODE_FACTORY(VolumetricLightNode);
} // namespace nimble