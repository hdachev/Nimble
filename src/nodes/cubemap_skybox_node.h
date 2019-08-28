#pragma once

#include "../render_node.h"

namespace nimble
{
class CubemapSkyboxNode : public RenderNode
{
public:
    CubemapSkyboxNode(RenderGraph* graph);
    ~CubemapSkyboxNode();

    void        declare_connections() override;
    bool        initialize(Renderer* renderer, ResourceManager* res_mgr) override;
    void        execute(double delta, Renderer* renderer, Scene* scene, View* view) override;
    void        shutdown() override;
    std::string name() override;

private:
    std::shared_ptr<RenderTarget> m_color_rt;
    std::shared_ptr<RenderTarget> m_depth_rt;
    RenderTargetView              m_scene_rtv;
    RenderTargetView              m_depth_rtv;
    std::shared_ptr<Shader>       m_vs;
    std::shared_ptr<Shader>       m_fs;
    std::shared_ptr<Program>      m_program;
};

DECLARE_RENDER_NODE_FACTORY(CubemapSkyboxNode);
} // namespace nimble
