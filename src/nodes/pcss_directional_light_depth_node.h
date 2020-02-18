#pragma once

#include "pcf_directional_light_depth_node.h"

namespace nimble
{
class PCSSDirectionalLightDepthNode : public PCFDirectionalLightDepthNode
{
public:
    PCSSDirectionalLightDepthNode(RenderGraph* graph);
    ~PCSSDirectionalLightDepthNode();

    std::string shadow_test_source_path() override;
    std::string name() override;
    std::vector<std::string> PCSSDirectionalLightDepthNode::shadow_test_source_defines() override;
};

DECLARE_RENDER_NODE_FACTORY(PCSSDirectionalLightDepthNode);
} // namespace nimble