#pragma once

#include "../render_node.h"

namespace nimble
{
class AdaptiveExposureNode : public RenderNode
{
public:
    AdaptiveExposureNode(RenderGraph* graph);
    ~AdaptiveExposureNode();

    void        declare_connections() override;
    bool        initialize(Renderer* renderer, ResourceManager* res_mgr) override;
    void        execute(double delta, Renderer* renderer, Scene* scene, View* view) override;
    void        shutdown() override;
    std::string name() override;

private:
    void initial_luminance(Renderer* renderer, Scene* scene, View* view);
    void compute_luminance(Renderer* renderer, Scene* scene, View* view);
    void average_luminance(double delta, Renderer* renderer, Scene* scene, View* view);

private:
    std::shared_ptr<RenderTarget> m_color_rt;
    std::shared_ptr<RenderTarget> m_luma_rt;
	std::shared_ptr<RenderTarget> m_compute_luma_rt;
	std::shared_ptr<RenderTarget> m_avg_luma_rt;

	RenderTargetView m_luma_rtv;

    std::shared_ptr<Shader> m_vs;

    std::shared_ptr<Shader>  m_lum_fs;
    std::shared_ptr<Program> m_lum_program;

    std::shared_ptr<Shader>  m_compute_lum_fs;
    std::shared_ptr<Program> m_compute_lum_program;

    std::shared_ptr<Shader>  m_average_lum_fs;
    std::shared_ptr<Program> m_average_lum_program;

    float   m_middle_grey = 0.18f;
    float   m_tau         = 1.1f;
};

DECLARE_RENDER_NODE_FACTORY(AdaptiveExposureNode);
} // namespace nimble