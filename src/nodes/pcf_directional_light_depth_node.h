#pragma once

#include "../render_node.h"

namespace nimble
{
class PCFDirectionalLightDepthNode : public RenderNode
{
public:
    PCFDirectionalLightDepthNode(RenderGraph* graph);
    ~PCFDirectionalLightDepthNode();

    void        declare_connections() override;
    bool        initialize(Renderer* renderer, ResourceManager* res_mgr) override;
    void        execute(double delta, Renderer* renderer, Scene* scene, View* view) override;
    void        shutdown() override;
    std::string name() override;

private:
    std::shared_ptr<GeometryShaderLibrary> m_library;
};

DECLARE_RENDER_NODE_FACTORY(PCFDirectionalLightDepthNode);
} // namespace nimble