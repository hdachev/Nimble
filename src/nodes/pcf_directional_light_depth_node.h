#pragma once

#include "../render_node.h"

namespace nimble
{
enum PCFFilteringTechnique
{
    PCF_FILTERING_GRID_9_SAMPLES,
    PCF_FILTERING_GRID_25_SAMPLES,
    PCF_FILTERING_GRID_49_SAMPLES,
    PCF_FILTERING_POISSON_8_SAMPLES,
    PCF_FILTERING_POISSON_16_SAMPLES,
    PCF_FILTERING_POISSON_32_SAMPLES,
    PCF_FILTERING_POISSON_64_SAMPLES
};

class PCFLightDepthNode : public RenderNode
{
public:
    PCFLightDepthNode(RenderGraph* graph);
    ~PCFLightDepthNode();

    void declare_connections() override;
    bool initialize(Renderer* renderer, ResourceManager* res_mgr) override;
    void execute(double delta, Renderer* renderer, Scene* scene, View* view) override;
    void shutdown() override;

protected:
    std::shared_ptr<GeneratedShaderLibrary> m_library;
    PCFFilteringTechnique                   m_filtering_technique = PCF_FILTERING_GRID_49_SAMPLES;
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