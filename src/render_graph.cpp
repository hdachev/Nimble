#include "render_graph.h"
#include "renderer.h"
#include "utility.h"
#include "logger.h"

namespace nimble
{
// -----------------------------------------------------------------------------------------------------------------------------------

RenderGraph::RenderGraph() :
    m_num_cascade_views(0), m_manual_cascade_rendering(false), m_per_cascade_culling(true)
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

RenderGraph::~RenderGraph()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool RenderGraph::initialize(Renderer* renderer, ResourceManager* res_mgr)
{
    for (auto& node : m_flattened_graph)
    {
        if (!node->initialize(renderer, res_mgr))
            return false;
    }

    return true;
}

// -----------------------------------------------------------------------------------------------------------------------------------

RenderGraphType RenderGraph::type()
{
    return RENDER_GRAPH_STANDARD;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RenderGraph::shutdown()
{
    for (auto& node : m_flattened_graph)
        node->shutdown();

    clear();
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RenderGraph::clear()
{
    m_end_node.reset();
    m_flattened_graph.clear();
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RenderGraph::build(std::shared_ptr<RenderNode> end_node)
{
    m_end_node = end_node;
    flatten_graph();
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RenderGraph::execute(Renderer* renderer, Scene* scene, View* view)
{
    for (auto& node : m_flattened_graph)
    {
        node->execute(renderer, scene, view);

        if (m_num_cascade_views > 0)
        {
            for (uint32_t i = 0; i < m_num_cascade_views; i++)
            {
                View* light_view = m_cascade_views[i];

                if (light_view)
                {
                    if (light_view->graph)
                        light_view->graph->execute(renderer, scene, light_view);
                    else
                        NIMBLE_LOG_ERROR("Render Graph not assigned for View!");
                }
            }

            m_num_cascade_views = 0;
        }
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::shared_ptr<RenderNode> RenderGraph::node_by_name(const std::string& name)
{
    for (const auto& node : m_flattened_graph)
    {
        if (node->name() == name)
            return node;
    }

    return nullptr;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RenderGraph::trigger_cascade_view_render(View* view)
{
    m_num_cascade_views = view->num_cascade_views;

    for (uint32_t i = 0; i < m_num_cascade_views; i++)
        m_cascade_views[i] = view->cascade_views[i];
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RenderGraph::on_window_resized(const uint32_t& w, const uint32_t& h)
{
    m_window_width  = w;
    m_window_height = h;

    for (auto& node : m_flattened_graph)
        node->on_window_resized(w, h);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RenderGraph::flatten_graph()
{
    m_flattened_graph.clear();

    if (m_end_node)
        traverse_and_push_node(m_end_node);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RenderGraph::traverse_and_push_node(std::shared_ptr<RenderNode> node)
{
    auto& input_rts = node->input_render_targets();

    for (auto& con : input_rts)
    {
        if (con.prev_node)
            traverse_and_push_node(con.prev_node);
    }

    auto& input_buffers = node->input_buffers();

    for (auto& con : input_buffers)
    {
        if (con.prev_node)
            traverse_and_push_node(con.prev_node);
    }

    // If node hasn't been pushed already, push it
    if (!is_node_pushed(node))
        m_flattened_graph.push_back(node);
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool RenderGraph::is_node_pushed(std::shared_ptr<RenderNode> node)
{
    for (auto& c_node : m_flattened_graph)
    {
        if (c_node->name() == node->name())
            return true;
    }

    return false;
}

// -----------------------------------------------------------------------------------------------------------------------------------

ShadowRenderGraph::ShadowRenderGraph() :
    RenderGraph()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool ShadowRenderGraph::initialize(Renderer* renderer, ResourceManager* res_mgr)
{
    if (!RenderGraph::initialize(renderer, res_mgr))
        return false;

    std::string includes;
    std::string defines;

    if (!utility::read_shader_separate(utility::path_for_resource("assets/" + m_sampling_source_path), includes, m_sampling_source, defines))
    {
        NIMBLE_LOG_ERROR("Failed load Sampling Source: " + m_sampling_source_path);
        return false;
    }

    return true;
}

// -----------------------------------------------------------------------------------------------------------------------------------

RenderGraphType ShadowRenderGraph::type()
{
    return RENDER_GRAPH_SHADOW;
}

// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble