#pragma once

#include "../render_node.h"

namespace nimble
{
class TiledForwardNode : public RenderNode
{
public:
    TiledForwardNode(RenderGraph* graph);
    ~TiledForwardNode();

    void        declare_connections() override;
    bool        initialize(Renderer* renderer, ResourceManager* res_mgr) override;
    void        execute(double delta, Renderer* renderer, Scene* scene, View* view) override;
    void        shutdown() override;
    std::string name() override;

private:
    bool                                    m_visualize_heat_map = false;
    std::shared_ptr<GeneratedShaderLibrary> m_library;
    std::shared_ptr<RenderTarget>           m_color_rt;
    std::shared_ptr<RenderTarget>           m_depth_rt;
    std::shared_ptr<RenderTarget>           m_velocity_rt;
    std::shared_ptr<ComputeBuffer>          m_light_indices;
    std::shared_ptr<ComputeBuffer>          m_light_grid;
    RenderTargetView                        m_color_rtv[2];
    RenderTargetView                        m_depth_rtv;
};

DECLARE_RENDER_NODE_FACTORY(TiledForwardNode);
} // namespace nimble
