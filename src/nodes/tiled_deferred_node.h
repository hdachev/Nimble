#pragma once

#include "../render_node.h"

namespace nimble
{
class TiledDeferredNode : public RenderNode
{
public:
    TiledDeferredNode(RenderGraph* graph);
    ~TiledDeferredNode();

    void        declare_connections() override;
    bool        initialize(Renderer* renderer, ResourceManager* res_mgr) override;
    void        execute(double delta, Renderer* renderer, Scene* scene, View* view) override;
    void        shutdown() override;
    std::string name() override;

private:
    uint32_t                      m_flags;
    std::shared_ptr<RenderTarget> m_color_rt;
    RenderTargetView              m_color_rtv;
    std::shared_ptr<Shader>       m_vs;
    std::shared_ptr<Shader>       m_fs;
    std::shared_ptr<Program>      m_program;

    // Input
    std::shared_ptr<RenderTarget> m_gbuffer1_rt;
    std::shared_ptr<RenderTarget> m_gbuffer2_rt;
    std::shared_ptr<RenderTarget> m_gbuffer3_rt;
    std::shared_ptr<RenderTarget> m_gbuffer4_rt;
    std::shared_ptr<RenderTarget> m_ssao_rt;
    std::shared_ptr<RenderTarget> m_depth_rt;
    std::shared_ptr<ComputeBuffer> m_light_indices;
    std::shared_ptr<ComputeBuffer> m_light_grid;
};

DECLARE_RENDER_NODE_FACTORY(TiledDeferredNode);
} // namespace nimble
