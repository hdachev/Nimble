#include "pcf_directional_light_depth_node.h"
#include "../render_graph.h"
#include "../renderer.h"

namespace nimble
{
DEFINE_RENDER_NODE_FACTORY(PCFDirectionalLightDepthNode)

// -----------------------------------------------------------------------------------------------------------------------------------

PCFDirectionalLightDepthNode::PCFDirectionalLightDepthNode(RenderGraph* graph) :
    RenderNode(graph)
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

PCFDirectionalLightDepthNode::~PCFDirectionalLightDepthNode()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void PCFDirectionalLightDepthNode::declare_connections()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool PCFDirectionalLightDepthNode::initialize(Renderer* renderer, ResourceManager* res_mgr)
{
    m_library = renderer->shader_cache().load_geometry_shader_library("shader/shadows/directional_light/shadow_map/directional_light_depth_vs.glsl", "shader/shadows/directional_light/shadow_map/directional_light_depth_fs.glsl");
    return true;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void PCFDirectionalLightDepthNode::execute(double delta, Renderer* renderer, Scene* scene, View* view)
{
    int32_t w = 0;
    int32_t h = 0;

    if (view->dest_render_target_view->texture->target() == GL_TEXTURE_2D)
    {
        Texture2D* texture = (Texture2D*)view->dest_render_target_view->texture.get();

        w = texture->width();
        h = texture->height();
    }
    else
    {
        TextureCube* texture = (TextureCube*)view->dest_render_target_view->texture.get();

        w = texture->width();
        h = texture->height();
    }

    renderer->bind_render_targets(0, nullptr, view->dest_render_target_view);

    glViewport(0, 0, h, h);

    glEnable(GL_DEPTH_TEST);

    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    glClear(GL_DEPTH_BUFFER_BIT);

    render_scene(renderer, scene, view, m_library.get(), NODE_USAGE_SHADOW_MAP);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void PCFDirectionalLightDepthNode::shutdown()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::string PCFDirectionalLightDepthNode::name()
{
    return "PCF Directional Light Depth";
}

// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble