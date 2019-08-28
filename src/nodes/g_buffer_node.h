#pragma once

#include "../render_node.h"
#include "../generated_shader_library.h"

namespace nimble
{
class GBufferNode : public RenderNode
{
public:
    GBufferNode(RenderGraph* graph);
    ~GBufferNode();

    void        declare_connections() override;
    bool        initialize_private(Renderer* renderer, ResourceManager* res_mgr) override;
    void        execute(double delta, Renderer* renderer, Scene* scene, View* view) override;
    void        shutdown() override;
    std::string name() override;

private:
    std::shared_ptr<RenderTarget>          m_gbuffer1_rt; // RGBA8 =  RGB: Albedo, A: -
    std::shared_ptr<RenderTarget>          m_gbuffer2_rt; // RGBA32F = RGB: Normal
    std::shared_ptr<RenderTarget>          m_gbuffer3_rt; // RGBA16F = RG: Velocity (Packed as 16x2)
    std::shared_ptr<RenderTarget>          m_gbuffer4_rt; // RGBA8 = R: Metallic, G: Roughness, B: Displacement, A: -
    std::shared_ptr<RenderTarget>          m_depth_rt;
    std::shared_ptr<GeneratedShaderLibrary> m_library;
    RenderTargetView                       m_gbuffer_rtv[4];
    RenderTargetView                       m_depth_rtv;
};

DECLARE_RENDER_NODE_FACTORY(GBufferNode);
} // namespace nimble
