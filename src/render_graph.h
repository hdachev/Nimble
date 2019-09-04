#pragma once

#include "render_node.h"
#include "view.h"

namespace nimble
{
class Renderer;
class ResourceManager;

class RenderGraph
{
public:
    RenderGraph();
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
} // namespace nimble