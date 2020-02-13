#include "clustered_deferred_node.h"
#include "clustered_light_culling_node.h"
#include "../render_graph.h"
#include "../resource_manager.h"
#include "../renderer.h"
#include "../logger.h"

namespace nimble
{
DEFINE_RENDER_NODE_FACTORY(ClusteredDeferredNode)

// -----------------------------------------------------------------------------------------------------------------------------------

ClusteredDeferredNode::ClusteredDeferredNode(RenderGraph* graph) :
    RenderNode(graph)
{
    m_flags = NODE_USAGE_PER_VIEW_UBO | NODE_USAGE_POINT_LIGHTS | NODE_USAGE_SPOT_LIGHTS | NODE_USAGE_DIRECTIONAL_LIGHTS | NODE_USAGE_SHADOW_MAPPING | NODE_USAGE_STATIC_MESH | NODE_USAGE_SKELETAL_MESH;
}

// -----------------------------------------------------------------------------------------------------------------------------------

ClusteredDeferredNode::~ClusteredDeferredNode()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void ClusteredDeferredNode::declare_connections()
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

bool ClusteredDeferredNode::initialize(Renderer* renderer, ResourceManager* res_mgr)
{
    register_bool_parameter("Visualize Heat Map", m_visualize_heat_map);

    m_gbuffer1_rt = find_input_render_target("G-Buffer1");
    m_gbuffer2_rt = find_input_render_target("G-Buffer2");
    m_gbuffer3_rt = find_input_render_target("G-Buffer3");
    m_gbuffer4_rt = find_input_render_target("G-Buffer4");
    m_ssao_rt     = find_input_render_target("SSAO");
    m_depth_rt    = find_input_render_target("Depth");

    m_light_indices = find_input_buffer("LightIndices");
    m_light_grid = find_input_buffer("LightGrid");

    m_cluster_data = std::shared_ptr<UniformBuffer>(new UniformBuffer(GL_DYNAMIC_DRAW, sizeof(glm::vec4)));

    m_color_rtv = RenderTargetView(0, 0, 0, m_color_rt->texture);

    m_vs = res_mgr->load_shader("shader/post_process/fullscreen_triangle_vs.glsl", GL_VERTEX_SHADER);
    m_fs = res_mgr->load_shader("shader/deferred/clustered_deferred_fs.glsl", GL_FRAGMENT_SHADER, m_flags, renderer);

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

void ClusteredDeferredNode::execute(double delta, Renderer* renderer, Scene* scene, View* view)
{
    uint32_t tile_size_x = (float(m_graph->actual_viewport_width() + CLUSTER_GRID_DIM_X - 1) / float(CLUSTER_GRID_DIM_X));
    uint32_t tile_size_y = (float(m_graph->actual_viewport_height() + CLUSTER_GRID_DIM_Y - 1) / float(CLUSTER_GRID_DIM_Y));

    uint32_t largest_tile_extent = std::max(tile_size_x, tile_size_y);

    glm::vec4 cluster_data = glm::vec4((float)CLUSTER_GRID_DIM_Z / std::log2f(view->far_plane / view->near_plane),
                                       -((float)CLUSTER_GRID_DIM_Z * std::log2f(view->near_plane) / std::log2f(view->far_plane / view->near_plane)),
                                       1.0f / float(largest_tile_extent),
                                       (float)m_visualize_heat_map);

    void* ptr = m_cluster_data->map(GL_WRITE_ONLY);

    memcpy(ptr, &cluster_data, sizeof(cluster_data));

    m_cluster_data->unmap();

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
    m_cluster_data->bind_base(5);

    render_fullscreen_triangle(renderer, view, m_program.get(), tex_unit, m_flags);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void ClusteredDeferredNode::shutdown()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::string ClusteredDeferredNode::name()
{
    return "Clustered Deferred";
}

// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble