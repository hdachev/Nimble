#pragma once

#include "../render_node.h"

namespace nimble
{
class ColorGradeNode : public RenderNode
{
public:
    ColorGradeNode(RenderGraph* graph);
    ~ColorGradeNode();

    void        declare_connections() override;
    bool        initialize(Renderer* renderer, ResourceManager* res_mgr) override;
    void        execute(double delta, Renderer* renderer, Scene* scene, View* view) override;
    void        shutdown() override;
    std::string name() override;

private:
    std::unique_ptr<Texture3D>    m_lut;
    std::shared_ptr<Shader>       m_vs;
    std::shared_ptr<Shader>       m_fs;
    std::shared_ptr<Program>      m_program;
    std::shared_ptr<RenderTarget> m_texture;
    std::shared_ptr<RenderTarget> m_output_rt;
    RenderTargetView              m_output_rtv;
};

DECLARE_RENDER_NODE_FACTORY(ColorGradeNode);
} // namespace nimble
