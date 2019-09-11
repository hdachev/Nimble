#pragma once

#include "../render_node.h"

namespace nimble
{
class VignetteNode : public RenderNode
{
public:
    VignetteNode(RenderGraph* graph);
    ~VignetteNode();

    void        declare_connections() override;
    bool        initialize(Renderer* renderer, ResourceManager* res_mgr) override;
    void        execute(double delta, Renderer* renderer, Scene* scene, View* view) override;
    void        shutdown() override;
    std::string name() override;
};

DECLARE_RENDER_NODE_FACTORY(VignetteNode);
    } // namespace nimble
