#include "tiled_deferred_node.h"
#include "../render_graph.h"
#include "../resource_manager.h"
#include "../renderer.h"
#include "../logger.h"

namespace nimble
{
DEFINE_RENDER_NODE_FACTORY(TiledDeferredNode)

// -----------------------------------------------------------------------------------------------------------------------------------

TiledDeferredNode::TiledDeferredNode(RenderGraph* graph) :
    RenderNode(graph)
{
    m_flags = NODE_USAGE_PER_VIEW_UBO | NODE_USAGE_POINT_LIGHTS | NODE_USAGE_SPOT_LIGHTS | NODE_USAGE_DIRECTIONAL_LIGHTS | NODE_USAGE_SHADOW_MAPPING | NODE_USAGE_STATIC_MESH | NODE_USAGE_SKELETAL_MESH;
}

// -----------------------------------------------------------------------------------------------------------------------------------

TiledDeferredNode::~TiledDeferredNode()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void TiledDeferredNode::declare_connections()
{
    // Declare the inputs to this render node
    register_input_render_target("G-Buffer1");
    register_input_render_target("G-Buffer2");
    register_input_render_target("G-Buffer3");
    register_input_render_target("G-Buffer4");
    register_input_render_target("SSAO");
    register_input_render_target("Depth");

    register_input_buffer("LightIndices");
    register_input_buffer("LightGrid");

    // Since we're rendering to the render targets provided as input, we'll simply forward the input
    // render targets as outputs.
    m_color_rt = register_scaled_output_render_target("Color", 1.0f, 1.0f, GL_TEXTURE_2D, GL_RGBA16F, GL_RGBA, GL_HALF_FLOAT, 1, 1, 1);
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool TiledDeferredNode::initialize(Renderer* renderer, ResourceManager* res_mgr)
{
    m_gbuffer1_rt = find_input_render_target("G-Buffer1");
    m_gbuffer2_rt = find_input_render_target("G-Buffer2");
    m_gbuffer3_rt = find_input_render_target("G-Buffer3");
    m_gbuffer4_rt = find_input_render_target("G-Buffer4");
    m_ssao_rt     = find_input_render_target("SSAO");
    m_depth_rt    = find_input_render_target("Depth");

    m_light_indices = find_input_buffer("LightIndices");
    m_light_grid = find_input_buffer("LightGrid");

    m_color_rtv = RenderTargetView(0, 0, 0, m_color_rt->texture);

    m_vs = res_mgr->load_shader("shader/post_process/fullscreen_triangle_vs.glsl", GL_VERTEX_SHADER);
    m_fs = res_mgr->load_shader("shader/deferred/tiled_deferred_fs.glsl", GL_FRAGMENT_SHADER, m_flags, renderer);

    if (m_vs && m_fs)
    {
        m_program = renderer->create_program(m_vs, m_fs);

        if (m_program)
            return true;
        else
        {
            NIMBLE_LOG_ERROR("Failed to create Program!");
            return false;
        }
    }
    else
    {
        NIMBLE_LOG_ERROR("Failed to load Shaders!");
        return false;
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void TiledDeferredNode::execute(double delta, Renderer* renderer, Scene* scene, View* view)
{
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    m_program->use();

    renderer->bind_render_targets(1, &m_color_rtv, nullptr);
    glClear(GL_COLOR_BUFFER_BIT);
    glViewport(0, 0, m_graph->window_width(), m_graph->window_height());

    int32_t tex_unit = 0;

    if (m_program->set_uniform("s_GBufferRT1", tex_unit) && m_gbuffer1_rt)
        m_gbuffer1_rt->texture->bind(tex_unit++);

    if (m_program->set_uniform("s_GBufferRT2", tex_unit) && m_gbuffer2_rt)
        m_gbuffer2_rt->texture->bind(tex_unit++);

    if (m_program->set_uniform("s_GBufferRT3", tex_unit) && m_gbuffer3_rt)
        m_gbuffer3_rt->texture->bind(tex_unit++);

    if (m_program->set_uniform("s_GBufferRT4", tex_unit) && m_gbuffer4_rt)
        m_gbuffer4_rt->texture->bind(tex_unit++);

    if (m_program->set_uniform("s_Depth", tex_unit) && m_depth_rt)
        m_depth_rt->texture->bind(tex_unit++);

    if (m_program->set_uniform("s_SSAO", tex_unit) && m_depth_rt)
        m_ssao_rt->texture->bind(tex_unit++);

    m_light_indices->buffer->bind_base(3);
    m_light_grid->buffer->bind_base(4);

    render_fullscreen_triangle(renderer, view, m_program.get(), tex_unit, m_flags);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void TiledDeferredNode::shutdown()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::string TiledDeferredNode::name()
{
    return "Tiled Deferred";
}

// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble