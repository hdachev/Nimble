#include "tiled_forward_node.h"
#include "../render_graph.h"
#include "../renderer.h"

namespace nimble
{
DEFINE_RENDER_NODE_FACTORY(TiledForwardNode)

// -----------------------------------------------------------------------------------------------------------------------------------

TiledForwardNode::TiledForwardNode(RenderGraph* graph) :
    RenderNode(graph)
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

TiledForwardNode::~TiledForwardNode()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void TiledForwardNode::declare_connections()
{
    register_input_buffer("LightIndices");

    m_color_rt    = register_scaled_output_render_target("Color", 1.0f, 1.0f, GL_TEXTURE_2D, GL_RGBA16F, GL_RGBA, GL_HALF_FLOAT);
    m_depth_rt    = register_scaled_output_render_target("Depth", 1.0f, 1.0f, GL_TEXTURE_2D, GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT);
    m_velocity_rt = register_scaled_output_render_target("Velocity", 1.0f, 1.0f, GL_TEXTURE_2D, GL_RG16F, GL_RG, GL_HALF_FLOAT);
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool TiledForwardNode::initialize(Renderer* renderer, ResourceManager* res_mgr)
{
    m_light_indices = find_input_buffer("LightIndices");

    m_library = renderer->shader_cache().load_generated_library("shader/forward/forward_vs.glsl", "shader/forward/tiled_forward_fs.glsl");

    m_color_rtv[0] = RenderTargetView(0, 0, 0, m_color_rt->texture);
    m_color_rtv[1] = RenderTargetView(0, 0, 0, m_velocity_rt->texture);
    m_depth_rtv    = RenderTargetView(0, 0, 0, m_depth_rt->texture);

    return true;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void TiledForwardNode::execute(double delta, Renderer* renderer, Scene* scene, View* view)
{
    renderer->bind_render_targets(2, m_color_rtv, &m_depth_rtv);
    glViewport(0, 0, m_graph->window_width(), m_graph->window_height());

    glEnable(GL_DEPTH_TEST);

    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    m_light_indices->buffer->bind_base(3);

    render_scene(renderer, scene, view, m_library.get(), NODE_USAGE_DEFAULT);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void TiledForwardNode::shutdown()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::string TiledForwardNode::name()
{
    return "Tiled Forward";
}

// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble