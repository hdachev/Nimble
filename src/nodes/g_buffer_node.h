#pragma once

#include "../render_node.h"

namespace nimble
{
class GBufferNode : public SceneRenderNode
{
public:
    GBufferNode(RenderGraph* graph);
    ~GBufferNode();

    bool        register_resources() override;
    bool        initialize() override;
    void        shutdown() override;
    std::string name() override;
    std::string vs_template_path() override;
    std::string fs_template_path() override;

protected:
    void execute_internal(const View* view) override;

private:
    std::shared_ptr<RenderTarget> m_gbuffer1_rt; // RGBA8 =  RGB: Albedo, A: -
	std::shared_ptr<RenderTarget> m_gbuffer2_rt; // RGBA32F = RGB: Normal, A: Velocity (Packed as 16x2)
	std::shared_ptr<RenderTarget> m_gbuffer3_rt; // RGBA8 = R: Metallic, G: Roughness, B: Displacement, A: -
    std::shared_ptr<RenderTarget> m_depth_rt;
    std::shared_ptr<RenderTarget> m_velocity_rt;
    RenderTargetView              m_gbuffer1_rtv;
    RenderTargetView              m_gbuffer2_rtv;
	RenderTargetView              m_gbuffer3_rtv;
    RenderTargetView              m_depth_rtv;
};

DECLARE_RENDER_NODE_FACTORY(GBufferNode);
} // namespace nimble#pragma once
