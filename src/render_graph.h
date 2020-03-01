#pragma once

#include "render_node.h"
#include "view.h"
#include "static_hash_map.h"
#include "linear_allocator.h"

namespace nimble
{
class Renderer;
class ResourceManager;

class RenderGraph
{
public:
    RenderGraph(uint32_t w, uint32_t h);
    ~RenderGraph();

    void                          build(std::shared_ptr<RenderNode> end_node);
    void                          execute(double delta, Renderer* renderer, Scene* scene, View* view);
    void                          shutdown();
    void                          clear();
    std::shared_ptr<RenderNode>   node_by_name(const std::string& name);
    void                          trigger_cascade_view_render(View* view);
    std::shared_ptr<RenderTarget> output_render_target();
    void                          on_window_resized(const uint32_t& w, const uint32_t& h);

    inline void                        set_name(const std::string& name) { m_name = name; }
    inline void                        set_is_shadow(bool shadow) { m_is_shadow = shadow; }
    inline bool                        is_shadow() { return m_is_shadow; }
    inline std::string                 name() { return m_name; }
    inline uint32_t                    node_count() { return (uint32_t)m_flattened_graph.size(); }
    inline std::shared_ptr<RenderNode> node(const uint32_t& idx) { return m_flattened_graph[idx]; }
    inline uint32_t                    window_width() { return m_window_width; }
    inline uint32_t                    window_height() { return m_window_height; }
    inline virtual uint32_t            actual_viewport_width() { return window_width(); }
    inline virtual uint32_t            actual_viewport_height() { return window_height(); }
    inline virtual uint32_t            rendered_viewport_width() { return actual_viewport_width(); }
    inline virtual uint32_t            rendered_viewport_height() { return actual_viewport_height(); }
    inline void                        set_manual_cascade_rendering(bool value) { m_manual_cascade_rendering = value; }
    inline bool                        is_manual_cascade_rendering() { return m_manual_cascade_rendering; }
    inline void                        set_per_cascade_culling(bool value) { m_per_cascade_culling = value; }
    inline bool                        per_cascade_culling() { return m_per_cascade_culling; }

    virtual bool initialize(Renderer* renderer, ResourceManager* res_mgr);

private:
    void flatten_graph();
    void traverse_and_push_node(std::shared_ptr<RenderNode> node);
    bool is_node_pushed(std::shared_ptr<RenderNode> node);

protected:
    std::string                              m_name;
    uint32_t                                 m_window_width;
    uint32_t                                 m_window_height;
    bool                                     m_per_cascade_culling;
    bool                                     m_manual_cascade_rendering;
    uint32_t                                 m_num_cascade_views;
    View*                                    m_cascade_views[MAX_SHADOW_CASTING_DIRECTIONAL_LIGHTS * MAX_SHADOW_MAP_CASCADES];
    std::shared_ptr<RenderNode>              m_end_node;
    std::vector<std::shared_ptr<RenderNode>> m_flattened_graph;
    bool                                     m_is_shadow = false;
};

class RenderPass;
class RenderGraphNew;

struct RenderResourceID
{
    uint64_t m_id;

    RenderResourceID(std::string name = "");
};

struct RenderTextureResourceDesc
{
    uint32_t w;
    uint32_t h;
    GLenum   format;
};

struct RenderBufferResourceDesc
{
    size_t size;
    GLenum flags;
};

enum RenderResourceType
{
    RENDER_RESOURCE_TEXTURE,
    RENDER_RESOURCE_BUFFER
};

class RenderResource
{
public:
    friend class RenderPass;
    friend class RenderResourceManager;

    RenderResourceType type;
    Texture*           texture = nullptr;
    Buffer*            buffer  = nullptr;

private:
    RenderResourceID          id;
    RenderPass*               owner;
    uint32_t                  usage_count = 0;
    RenderPass*               usages[32];
    uint32_t                  usage_start = 0;
    uint32_t                  usage_end   = 0;
    RenderBufferResourceDesc  buffer_desc;
    RenderTextureResourceDesc texture_desc;

public:
    void add_usage(RenderPass* pass);
};

class RenderResourceManager
{
public:
    friend class RenderResource;

private:
    uint32_t                                       m_resource_count = 0;
    RenderResource                                 m_resources[1024];
    StaticHashMap<uint64_t, RenderResource*, 1024> m_resource_map;

public:
    RenderResourceManager();
    ~RenderResourceManager();

    RenderResource* add_output_buffer(const RenderResourceID& id, const RenderBufferResourceDesc& desc);
    RenderResource* add_output_texture(const RenderResourceID& id, const RenderTextureResourceDesc& desc);
    RenderResource* find_resource(const RenderResourceID& id);

    void realize_resource(RenderResource* res);
    void reclaim_resource(RenderResource* res);
};

class RenderPass
{
private:
    bool                                         m_render_to_backbuffer = false;
    StaticHashMap<uint64_t, RenderResource*, 32> m_inputs;
    StaticHashMap<uint64_t, RenderResource*, 32> m_outputs;

public:
    RenderPass();
    ~RenderPass();

    void render_to_backbuffer(bool backbuffer);
    bool is_render_to_backbuffer();

    RenderResource* add_buffer_dependency(const RenderResourceID& id, RenderResourceManager& manager);
    RenderResource* add_sampled_texture_dependency(const RenderResourceID& id, RenderResourceManager& manager);
    RenderResource* add_storage_texture_dependency(const RenderResourceID& id, RenderResourceManager& manager);

    RenderResource* add_output_buffer(const RenderResourceID& id, const RenderBufferResourceDesc& desc, RenderResourceManager& manager);
    RenderResource* add_output_texture(const RenderResourceID& id, const RenderTextureResourceDesc& desc, RenderResourceManager& manager);
    RenderResource* add_existing_output_texture(const RenderResourceID& id, RenderResourceManager& manager);

    virtual void execute();
};

template <typename T>
class RenderPassWithData : public RenderPass
{
public:
    friend class RenderGraphBuilder;
    T                                       m_pass_data;
    std::function<void(const T& pass_data)> m_execute_func;

public:
    void execute() override
    {
        m_execute_func(m_pass_data);
    }
};

class RenderGraphBuilder
{
private:
    RenderGraphNew& m_render_graph;

public:
    RenderGraphBuilder(RenderGraphNew& graph);
    ~RenderGraphBuilder();

    template <typename T>
    void add_render_pass(std::function<void(RenderPass& desc, T& pass_data, RenderResourceManager& manager)> setup,
                         std::function<void(const T& pass_data)>                                             execute,
                         RenderResourceManager&                                                              manager)
    {
        RenderPassWithData<T>* rp = m_render_graph.allocate_render_pass<T>();

        T pass_data;
        setup(*rp, pass_data, manager);

        rp->m_execute_func = execute;
        rp->m_pass_data    = pass_data;
    }
};

class RenderGraphNew
{
public:
    friend class RenderGraphBuilder;

private:
    RenderResourceManager            m_resource_manager;
    uint32_t                         m_render_pass_count = 0;
    void*                            m_render_pass_buffer;
    std::unique_ptr<LinearAllocator> m_render_pass_allocator;
    RenderPass*                      m_render_passes_allocated[256];
    RenderPass*                      m_render_passes_flattened[256];

public:
    RenderGraphNew();
    ~RenderGraphNew();

    void execute();

private:
    virtual void build(RenderGraphBuilder& builder, RenderResourceManager& manager) = 0;

    template <typename T>
    RenderPassWithData<T>* allocate_render_pass()
    {
        uint32_t               idx = m_render_pass_count++;
        RenderPassWithData<T>* rp = NIMBLE_NEW(m_render_pass_allocator.get()) RenderPassWithData<T>();
        m_render_passes_allocated[idx] = rp;

        return rp;
    }

    void flatten();
};
} // namespace nimble