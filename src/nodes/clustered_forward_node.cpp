#include "clustered_forward_node.h"
#include "clustered_light_culling_node.h"
#include "../render_graph.h"
#include "../renderer.h"

namespace nimble
{
DEFINE_RENDER_NODE_FACTORY(ClusteredForwardNode)

struct ClusterData
{
    glm::uvec4 cluster_size;
    glm::vec4  scale_bias;
};

// -----------------------------------------------------------------------------------------------------------------------------------

ClusteredForwardNode::ClusteredForwardNode(RenderGraph* graph) :
    RenderNode(graph)
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

ClusteredForwardNode::~ClusteredForwardNode()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void ClusteredForwardNode::declare_connections()
{
    register_input_render_target("Depth");
    register_input_buffer("LightIndices");
    register_input_buffer("LightGrid");

    m_color_rt    = register_scaled_output_render_target("Color", 1.0f, 1.0f, GL_TEXTURE_2D, GL_RGBA16F, GL_RGBA, GL_HALF_FLOAT);
    m_velocity_rt = register_scaled_output_render_target("Velocity", 1.0f, 1.0f, GL_TEXTURE_2D, GL_RG16F, GL_RG, GL_HALF_FLOAT);
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool ClusteredForwardNode::initialize(Renderer* renderer, ResourceManager* res_mgr)
{
    register_bool_parameter("Visualize Heat Map", m_visualize_heat_map);

    m_depth_rt = find_input_render_target("Depth");

    m_light_indices = find_input_buffer("LightIndices");
    m_light_grid = find_input_buffer("LightGrid");

    m_library = renderer->shader_cache().load_generated_library("shader/forward/forward_vs.glsl", "shader/forward/clustered_forward_fs.glsl");

    m_color_rtv[0] = RenderTargetView(0, 0, 0, m_color_rt->texture);
    m_color_rtv[1] = RenderTargetView(0, 0, 0, m_velocity_rt->texture);
    m_depth_rtv    = RenderTargetView(0, 0, 0, m_depth_rt->texture);

    m_cluster_data = std::shared_ptr<UniformBuffer>(new UniformBuffer(GL_DYNAMIC_STORAGE_BIT, sizeof(ClusterData)));

    return true;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void ClusteredForwardNode::execute(double delta, Renderer* renderer, Scene* scene, View* view)
{
    uint32_t tile_count_x  = ceil(float(m_graph->actual_viewport_width()) / float(CLUSTER_TILE_SIZE));
    uint32_t tile_count_y  = ceil(float(m_graph->actual_viewport_height()) / float(CLUSTER_TILE_SIZE));
    uint32_t grid_to_dim_y = (m_graph->actual_viewport_height() + tile_count_y - 1) / tile_count_y;
    float    sD            = 2.0f * tanf(glm::radians(view->fov) * 0.5f) / float(grid_to_dim_y);

    ClusterData cluster_data = 
    {
        glm::uvec4(tile_count_x, tile_count_y, CLUSTER_Z_SLICES, 0),
        glm::vec4(1.0f / view->near_plane, 1.0f / logf(sD + 1.0f), 0.0f, 0.0f)
    };

    void* ptr = m_cluster_data->map(GL_MAP_WRITE_BIT);

    memcpy(ptr, &cluster_data, sizeof(ClusterData));

    m_cluster_data->unmap();

    renderer->bind_render_targets(2, m_color_rtv, &m_depth_rtv);
    glViewport(0, 0, m_graph->window_width(), m_graph->window_height());

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_EQUAL);
    glDepthMask(GL_FALSE);

    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    m_light_indices->buffer->bind_base(3);
    m_light_grid->buffer->bind_base(4);
    m_cluster_data->bind_base(5);

    render_scene(renderer, scene, view, m_library.get(), NODE_USAGE_DEFAULT);

    glDepthFunc(GL_LESS);
    glDepthMask(GL_TRUE);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void ClusteredForwardNode::shutdown()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::string ClusteredForwardNode::name()
{
    return "Tiled Forward";
}

// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble