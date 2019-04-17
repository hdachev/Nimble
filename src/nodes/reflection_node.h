#pragma once

#include "../render_node.h"

namespace nimble
{
class ReflectionNode : public RenderNode
{
public:
    ReflectionNode(RenderGraph* graph);
    ~ReflectionNode();

    void        declare_connections() override;
    bool        initialize(Renderer* renderer, ResourceManager* res_mgr) override;
    void        execute(double delta, Renderer* renderer, Scene* scene, View* view) override;
    void        shutdown() override;
    std::string name() override;

private:
    // Inputs
    std::shared_ptr<RenderTarget> m_color_rt;
    std::shared_ptr<RenderTarget> m_ssr_rt;
 
    // Outputs
    std::shared_ptr<RenderTarget> m_reflection_rt;

    RenderTargetView m_reflection_rtv;

	std::shared_ptr<Shader>  m_fullscreen_triangle_vs;
    std::shared_ptr<Shader>  m_reflection_fs;
    std::shared_ptr<Program> m_reflection_program;
};

DECLARE_RENDER_NODE_FACTORY(ReflectionNode);
} // namespace nimble