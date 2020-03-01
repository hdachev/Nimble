#include "render_graph.h"
#include "renderer.h"
#include "utility.h"
#include "logger.h"
#include "profiler.h"

namespace nimble
{
// -----------------------------------------------------------------------------------------------------------------------------------

RenderGraph::RenderGraph(uint32_t w, uint32_t h) :
    m_num_cascade_views(0), m_manual_cascade_rendering(false), m_per_cascade_culling(true), m_window_width(w), m_window_height(h)
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

void RenderGraph::execute(double delta, Renderer* renderer, Scene* scene, View* view)
{
    for (auto& node : m_flattened_graph)
    {
        glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, node->name().c_str());

        {
            NIMBLE_SCOPED_SAMPLE(node->name().c_str());
            node->execute(delta, renderer, scene, view);
        }

        glPopDebugGroup();

        if (m_num_cascade_views > 0)
        {
            for (uint32_t i = 0; i < m_num_cascade_views; i++)
            {
                View* light_view = m_cascade_views[i];

                if (light_view)
                {
                    if (light_view->graph)
                        light_view->graph->execute(delta, renderer, scene, light_view);
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

std::shared_ptr<RenderTarget> RenderGraph::output_render_target()
{
    const auto& last_node = m_flattened_graph[m_flattened_graph.size() - 1];

    if (last_node->output_render_target_count() > 0)
        return last_node->output_render_target(0);

    return nullptr;
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

RenderResourceID::RenderResourceID(std::string name)
{
    m_id = NIMBLE_HASH(name.c_str());
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RenderResource::add_usage(RenderPass* pass)
{
    usages[usage_count] = pass;
    usage_count++;
}

// -----------------------------------------------------------------------------------------------------------------------------------

RenderResourceManager::RenderResourceManager()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

RenderResourceManager::~RenderResourceManager()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

RenderResource* RenderResourceManager::add_output_buffer(const RenderResourceID& id, const RenderBufferResourceDesc& desc)
{
    uint32_t idx = m_resource_count++;

    RenderResource* res = &m_resources[idx];

    res->id          = id;
    res->buffer_desc = desc;

    m_resource_map.set(id.m_id, res);

    return res;
}

// -----------------------------------------------------------------------------------------------------------------------------------

RenderResource* RenderResourceManager::add_output_texture(const RenderResourceID& id, const RenderTextureResourceDesc& desc)
{
    uint32_t idx = m_resource_count++;

    RenderResource* res = &m_resources[idx];

    res->id           = id;
    res->texture_desc = desc;

    m_resource_map.set(id.m_id, res);

    return res;
}

// -----------------------------------------------------------------------------------------------------------------------------------

RenderResource* RenderResourceManager::find_resource(const RenderResourceID& id)
{
    RenderResource* res = nullptr;

    m_resource_map.get(id.m_id, res);

    return res;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RenderResourceManager::realize_resource(RenderResource* res)
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RenderResourceManager::reclaim_resource(RenderResource* res)
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

RenderPass::RenderPass()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

RenderPass::~RenderPass()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RenderPass::render_to_backbuffer(bool backbuffer)
{
    m_render_to_backbuffer = backbuffer;
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool RenderPass::is_render_to_backbuffer()
{
    return m_render_to_backbuffer;
}

// -----------------------------------------------------------------------------------------------------------------------------------

RenderResource* RenderPass::add_buffer_dependency(const RenderResourceID& id, RenderResourceManager& manager)
{
    RenderResource* res = manager.find_resource(id);
    
    res->owner = this;
    res->add_usage(this);

    m_inputs.set(id.m_id, res);

    return res;
}

// -----------------------------------------------------------------------------------------------------------------------------------

RenderResource* RenderPass::add_sampled_texture_dependency(const RenderResourceID& id, RenderResourceManager& manager)
{
    RenderResource* res = manager.find_resource(id);

    res->owner = this;
    res->add_usage(this);

    m_inputs.set(id.m_id, res);

    return res;
}

// -----------------------------------------------------------------------------------------------------------------------------------

RenderResource* RenderPass::add_storage_texture_dependency(const RenderResourceID& id, RenderResourceManager& manager)
{
    RenderResource* res = manager.find_resource(id);

    res->owner = this;
    res->add_usage(this);

    m_inputs.set(id.m_id, res);

    return res;
}

// -----------------------------------------------------------------------------------------------------------------------------------

RenderResource* RenderPass::add_output_buffer(const RenderResourceID& id, const RenderBufferResourceDesc& desc, RenderResourceManager& manager)
{
    RenderResource* res = manager.add_output_buffer(id, desc);

    res->owner = this;
    m_outputs.set(id.m_id, res);

    return res;
}

// -----------------------------------------------------------------------------------------------------------------------------------

RenderResource* RenderPass::add_output_texture(const RenderResourceID& id, const RenderTextureResourceDesc& desc, RenderResourceManager& manager)
{
    RenderResource* res = manager.add_output_texture(id, desc);

    res->owner = this;
    m_outputs.set(id.m_id, res);

    return res;
}

// -----------------------------------------------------------------------------------------------------------------------------------

RenderResource* RenderPass::add_existing_output_texture(const RenderResourceID& id, RenderResourceManager& manager)
{
    // @TODO: Implement this!
    return nullptr;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RenderPass::execute()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

RenderGraphBuilder::RenderGraphBuilder(RenderGraphNew& graph) :
    m_render_graph(graph)
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

RenderGraphBuilder::~RenderGraphBuilder()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

RenderGraphNew::RenderGraphNew()
{
    // Allocate a small buffer for the LinearAllocator to use.
    size_t buffer_size      = 1024 * 1024;
    m_render_pass_buffer    = malloc(buffer_size);

    // Create LinearAllocatr for allocating dynamic sized RenderPassWithData<T> objects.
    m_render_pass_allocator = std::make_unique<LinearAllocator>(m_render_pass_buffer, buffer_size);
}

// -----------------------------------------------------------------------------------------------------------------------------------

RenderGraphNew::~RenderGraphNew()
{
    // Release render pass linear allocator
    m_render_pass_allocator.reset();

    // Free the buffer used in the allocator.
    free(m_render_pass_buffer);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RenderGraphNew::execute()
{
    // Clear linear allocator to begin allocating render passes for this frame.
    m_render_pass_allocator->clear();

    // Create render graph builder for this frame.
    RenderGraphBuilder builder(*this);

    // Build the render graph by calling the overriden build method of the child RenderGraph class.
    build(builder, m_resource_manager);

    // Flatten the previously built render graph in order to traverse it linearly and remove redundant passes.
    flatten();

    // Iterate over the flattened render graph and call each passes' execute method.
    for (uint32_t i = 0; i < m_render_pass_count; i++)
        m_render_passes_flattened[i]->execute();
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RenderGraphNew::flatten()
{

}

// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble