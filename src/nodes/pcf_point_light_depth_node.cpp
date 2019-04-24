#include "pcf_point_light_depth_node.h"
#include "../render_graph.h"
#include "../renderer.h"

namespace nimble
{
DEFINE_RENDER_NODE_FACTORY(PCFPointLightDepthNode)

// -----------------------------------------------------------------------------------------------------------------------------------

PCFPointLightDepthNode::PCFPointLightDepthNode(RenderGraph* graph) :
    RenderNode(graph)
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

PCFPointLightDepthNode::~PCFPointLightDepthNode()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void PCFPointLightDepthNode::execute(double delta, Renderer* renderer, Scene* scene, View* view)
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

    glViewport(0, 0, w, h);

    glEnable(GL_DEPTH_TEST);

    glEnable(GL_CULL_FACE);
    glCullFace(GL_FRONT);

    glClear(GL_DEPTH_BUFFER_BIT);

    render_scene(renderer, scene, view, m_library.get(), NODE_USAGE_SHADOW_MAP, std::bind(&PCFPointLightDepthNode::set_shader_uniforms, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
}

// -----------------------------------------------------------------------------------------------------------------------------------

void PCFPointLightDepthNode::declare_connections()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool PCFPointLightDepthNode::initialize(Renderer* renderer, ResourceManager* res_mgr)
{
    m_library = renderer->shader_cache().load_library("shader/shadows/point_light/shadow_map/point_light_depth_vs.glsl", "shader/shadows/point_light/shadow_map/point_light_depth_fs.glsl");
    return true;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void PCFPointLightDepthNode::shutdown()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void PCFPointLightDepthNode::set_shader_uniforms(View* view, Program* program, int32_t& tex_unit)
{
    program->set_uniform("u_LightIdx", static_cast<int32_t>(view->light_index));
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::string PCFPointLightDepthNode::name()
{
    return "PCF Point Light Depth";
}

// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble