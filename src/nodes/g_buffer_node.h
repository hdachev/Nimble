#pragma once

#include "../render_node.h"
#include "../shader_library.h"

namespace nimble
{
class GBufferNode : public RenderNode
{
public:
    GBufferNode(RenderGraph* graph);
    ~GBufferNode();

    void        declare_connections() override;
    bool        initialize(Renderer* renderer, ResourceManager* res_mgr) override;
    void        execute(Renderer* renderer, Scene* scene, View* view) override;
    void        shutdown() override;
    std::string name() override;

private:
    std::shared_ptr<RenderTarget>  m_gbuffer1_rt; // RGBA8 =  RGB: Albedo, A: -
    std::shared_ptr<RenderTarget>  m_gbuffer2_rt; // RGBA32F = RGB: Normal, A: Velocity (Packed as 16x2)
    std::shared_ptr<RenderTarget>  m_gbuffer3_rt; // RGBA8 = R: Metallic, G: Roughness, B: Displacement, A: -
    std::shared_ptr<RenderTarget>  m_depth_rt;
    std::shared_ptr<RenderTarget>  m_velocity_rt;
    std::shared_ptr<ShaderLibrary> m_library;
    RenderTargetView               m_gbuffer_rtv[3];
    RenderTargetView               m_depth_rtv;
};

DECLARE_RENDER_NODE_FACTORY(GBufferNode);
} // namespace nimble
