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
    register_enum_parameter("Filtering Technique", &m_filtering_technique, {
                                                                               { PCF_FILTERING_GRID_9_SAMPLES, "3x3 Grid (9 Samples)" },
                                                                               { PCF_FILTERING_GRID_25_SAMPLES, "5x5 Grid (25 Samples)" },
                                                                               { PCF_FILTERING_GRID_49_SAMPLES, "7x7 Grid (49 Samples)" },
                                                                               { PCF_FILTERING_POISSON_8_SAMPLES, "Poisson (8 Samples)" },
                                                                               { PCF_FILTERING_POISSON_16_SAMPLES, "Poisson (16 Samples)" },
                                                                               { PCF_FILTERING_POISSON_32_SAMPLES, "Poisson (32 Samples)" },
                                                                               { PCF_FILTERING_POISSON_64_SAMPLES, "Poisson (64 Samples)" },
                                                                           });
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