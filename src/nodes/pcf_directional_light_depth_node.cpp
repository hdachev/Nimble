#include "pcf_directional_light_depth_node.h"
#include "../render_graph.h"
#include "../renderer.h"

namespace nimble
{
DEFINE_RENDER_NODE_FACTORY(PCFDirectionalLightDepthNode)
DEFINE_RENDER_NODE_FACTORY(PCFSpotLightDepthNode)

// -----------------------------------------------------------------------------------------------------------------------------------

PCFLightDepthNode::PCFLightDepthNode(RenderGraph* graph) :
    RenderNode(graph)
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

PCFLightDepthNode::~PCFLightDepthNode()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void PCFLightDepthNode::declare_connections()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool PCFLightDepthNode::initialize(Renderer* renderer, ResourceManager* res_mgr)
{
    m_library = renderer->shader_cache().load_generated_library("shader/shadows/directional_light/shadow_map/directional_light_depth_vs.glsl", "shader/shadows/directional_light/shadow_map/directional_light_depth_fs.glsl");
    return true;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void PCFLightDepthNode::execute(double delta, Renderer* renderer, Scene* scene, View* view)
{
    int32_t w = 0;
    int32_t h = 0;

    if (view->dest_depth_render_target_view->texture->target() == GL_TEXTURE_2D)
    {
        Texture2D* texture = (Texture2D*)view->dest_depth_render_target_view->texture.get();

        w = texture->width();
        h = texture->height();
    }
    else
    {
        TextureCube* texture = (TextureCube*)view->dest_depth_render_target_view->texture.get();

        w = texture->width();
        h = texture->height();
    }

    renderer->bind_render_targets(0, nullptr, view->dest_depth_render_target_view);

    glViewport(0, 0, h, h);

    glEnable(GL_DEPTH_TEST);

    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    glClear(GL_DEPTH_BUFFER_BIT);

    render_scene(renderer, scene, view, m_library.get(), NODE_USAGE_SHADOW_MAP);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void PCFLightDepthNode::shutdown()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

PCFDirectionalLightDepthNode::PCFDirectionalLightDepthNode(RenderGraph* graph) :
    PCFLightDepthNode(graph)
{

}

// -----------------------------------------------------------------------------------------------------------------------------------

PCFDirectionalLightDepthNode::~PCFDirectionalLightDepthNode()
{

}

// -----------------------------------------------------------------------------------------------------------------------------------

std::string PCFDirectionalLightDepthNode::shadow_test_source_path()
{
    return "shader/shadows/directional_light/sampling/pcf_directional_light.glsl";
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::string PCFDirectionalLightDepthNode::name()
{
    return "PCF Directional Light Depth Node";
}

// -----------------------------------------------------------------------------------------------------------------------------------

PCFSpotLightDepthNode::PCFSpotLightDepthNode(RenderGraph* graph) :
    PCFLightDepthNode(graph)
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

PCFSpotLightDepthNode::~PCFSpotLightDepthNode()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::string PCFSpotLightDepthNode::shadow_test_source_path()
{
    return "shader/shadows/spot_light/sampling/pcf_spot_light.glsl";
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::string PCFSpotLightDepthNode::name()
{
    return "PCF Spot Light Depth Node";
}
// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble