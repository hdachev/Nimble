#pragma once

#include "../render_node.h"

namespace nimble
{
class TiledLightCullingNode : public RenderNode
{
public:
    TiledLightCullingNode(RenderGraph* graph);
    ~TiledLightCullingNode();

    void        declare_connections() override;
    bool        initialize(Renderer* renderer, ResourceManager* res_mgr) override;
    void        execute(double delta, Renderer* renderer, Scene* scene, View* view) override;
    void        shutdown() override;
    std::string name() override;

private:

};

DECLARE_RENDER_NODE_FACTORY(TiledLightCullingNode);
} // namespace nimble
