#pragma once

#include "../render_node.h"

namespace nimble
{
class PCFLightDepthNode : public RenderNode
{
public:
    PCFLightDepthNode(RenderGraph* graph);
    ~PCFLightDepthNode();

    void declare_connections() override;
    bool initialize(Renderer* renderer, ResourceManager* res_mgr) override;
    void execute(double delta, Renderer* renderer, Scene* scene, View* view) override;
    void shutdown() override;

private:
    std::shared_ptr<GeneratedShaderLibrary> m_library;
};

class PCFDirectionalLightDepthNode : public PCFLightDepthNode
{
public:
    PCFDirectionalLightDepthNode(RenderGraph* graph);
    ~PCFDirectionalLightDepthNode();

    std::string shadow_test_source_path() override;
    std::string name() override;
};

DECLARE_RENDER_NODE_FACTORY(PCFDirectionalLightDepthNode);

class PCFSpotLightDepthNode : public PCFLightDepthNode
{
public:
    PCFSpotLightDepthNode(RenderGraph* graph);
    ~PCFSpotLightDepthNode();

    std::string shadow_test_source_path() override;
    std::string name() override;
};

DECLARE_RENDER_NODE_FACTORY(PCFSpotLightDepthNode);
} // namespace nimble