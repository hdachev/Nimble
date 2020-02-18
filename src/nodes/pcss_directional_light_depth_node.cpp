#include "pcss_directional_light_depth_node.h"
#include "../render_graph.h"
#include "../renderer.h"

namespace nimble
{
DEFINE_RENDER_NODE_FACTORY(PCSSDirectionalLightDepthNode)

// -----------------------------------------------------------------------------------------------------------------------------------

PCSSDirectionalLightDepthNode::PCSSDirectionalLightDepthNode(RenderGraph* graph) :
    PCFDirectionalLightDepthNode(graph)
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

PCSSDirectionalLightDepthNode::~PCSSDirectionalLightDepthNode()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::string PCSSDirectionalLightDepthNode::shadow_test_source_path()
{
    return "shader/shadows/directional_light/sampling/pcss_directional_light.glsl";
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::string PCSSDirectionalLightDepthNode::name()
{
    return "PCSS Directional Light Depth Node";
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::vector<std::string> PCSSDirectionalLightDepthNode::shadow_test_source_defines()
{
    return { };
}

// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble