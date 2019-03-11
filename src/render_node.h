#pragma once

#include <functional>
#include <memory>

#include "render_target.h"
#include "view.h"
#include "macros.h"

#define REGISTER_RENDER_NODE(class_name, resource_manager) resource_manager.register_render_node_factory(#class_name, create_render_node_##class_name)
#define DECLARE_RENDER_NODE_FACTORY(class_name) extern std::shared_ptr<RenderNode> create_render_node_##class_name(RenderGraph* graph)
#define DEFINE_RENDER_NODE_FACTORY(class_name)                                      \
    std::shared_ptr<RenderNode> create_render_node_##class_name(RenderGraph* graph) \
    {                                                                               \
        auto node = std::make_shared<class_name>(graph);                            \
        node->declare_connections();                                                \
        return node;                                                                \
    }

namespace nimble
{
struct View;
struct FramebufferGroup;
class RenderGraph;
class Scene;
class ShaderLibrary;
class ResourceManager;
class Renderer;

enum RenderNodeFlags
{
    NODE_USAGE_PER_OBJECT_UBO        = BIT_FLAG(0),
    NODE_USAGE_PER_VIEW_UBO          = BIT_FLAG(1),
    NODE_USAGE_POINT_LIGHTS          = BIT_FLAG(2),
    NODE_USAGE_SPOT_LIGHTS           = BIT_FLAG(3),
    NODE_USAGE_DIRECTIONAL_LIGHTS    = BIT_FLAG(4),
    NODE_USAGE_SHADOW_MAPPING        = BIT_FLAG(5),
    NODE_USAGE_STATIC_MESH           = BIT_FLAG(6),
    NODE_USAGE_SKELETAL_MESH         = BIT_FLAG(7),
    NODE_USAGE_MATERIAL_ALBEDO       = BIT_FLAG(8),
    NODE_USAGE_MATERIAL_NORMAL       = BIT_FLAG(9),
    NODE_USAGE_MATERIAL_METAL_SPEC   = BIT_FLAG(10),
    NODE_USAGE_MATERIAL_ROUGH_SMOOTH = BIT_FLAG(11),
    NODE_USAGE_MATERIAL_DISPLACEMENT = BIT_FLAG(12),
    NODE_USAGE_MATERIAL_EMISSIVE     = BIT_FLAG(13),
    NODE_USAGE_ALL_MATERIALS         = NODE_USAGE_MATERIAL_ALBEDO | NODE_USAGE_MATERIAL_NORMAL | NODE_USAGE_MATERIAL_METAL_SPEC | NODE_USAGE_MATERIAL_ROUGH_SMOOTH | NODE_USAGE_MATERIAL_EMISSIVE | NODE_USAGE_MATERIAL_DISPLACEMENT,
    NODE_USAGE_DEFAULT               = NODE_USAGE_PER_OBJECT_UBO | NODE_USAGE_PER_VIEW_UBO | NODE_USAGE_POINT_LIGHTS | NODE_USAGE_SPOT_LIGHTS | NODE_USAGE_DIRECTIONAL_LIGHTS | NODE_USAGE_STATIC_MESH | NODE_USAGE_SKELETAL_MESH | NODE_USAGE_ALL_MATERIALS | NODE_USAGE_SHADOW_MAPPING,
    NODE_USAGE_SHADOW_MAP            = NODE_USAGE_PER_OBJECT_UBO | NODE_USAGE_PER_VIEW_UBO | NODE_USAGE_STATIC_MESH | NODE_USAGE_SKELETAL_MESH | NODE_USAGE_MATERIAL_ALBEDO
};

struct BoolParameter
{
    bool*       ptr;
    std::string name;
};

struct IntParameter
{
    int32_t*    ptr;
    std::string name;
};

struct FloatParameter
{
    float*      ptr;
    std::string name;
};

class RenderNode
{
public:
    struct OutputRenderTarget
    {
        std::string                   slot_name;
        std::shared_ptr<RenderTarget> render_target;
    };

    struct InputRenderTarget
    {
        std::string                   slot_name;
        std::string                   prev_slot_name;
        std::shared_ptr<RenderTarget> prev_render_target;
        std::shared_ptr<RenderNode>   prev_node;
    };

    struct OutputBuffer
    {
        std::string                          slot_name;
        std::shared_ptr<ShaderStorageBuffer> buffer;
    };

    struct InputBuffer
    {
        std::string                          slot_name;
        std::string                          prev_slot_name;
        std::shared_ptr<ShaderStorageBuffer> prev_buffer;
        std::shared_ptr<RenderNode>          prev_node;
    };

    RenderNode(RenderGraph* graph);
    ~RenderNode();

    std::shared_ptr<RenderTarget>        find_output_render_target(const std::string& name);
    std::shared_ptr<RenderTarget>        find_intermediate_render_target(const std::string& name);
    std::shared_ptr<RenderTarget>        find_input_render_target(const std::string& name);
    std::shared_ptr<ShaderStorageBuffer> find_output_buffer(const std::string& name);
    std::shared_ptr<ShaderStorageBuffer> find_input_buffer(const std::string& name);
    OutputRenderTarget*                  find_output_render_target_slot(const std::string& name);
    InputRenderTarget*                   find_input_render_target_slot(const std::string& name);
    OutputBuffer*                        find_output_buffer_slot(const std::string& name);
    InputBuffer*                         find_input_buffer_slot(const std::string& name);
    void                                 set_input(const std::string& name, OutputRenderTarget* render_target, std::shared_ptr<RenderNode> owner);
    void                                 set_input(const std::string& name, OutputBuffer* buffer, std::shared_ptr<RenderNode> owner);
    void                                 timing_total(float& cpu_time, float& gpu_time);
    void                                 set_bool_parameter(const std::string& name, bool value);
    void                                 set_int_parameter(const std::string& name, int32_t value);
    void                                 set_float_parameter(const std::string& name, float value);
    BoolParameter*                       bool_parameters(int32_t& count);
    IntParameter*                        int_parameters(int32_t& count);
    FloatParameter*                      float_parameters(int32_t& count);

    // Inline getters
    inline std::vector<InputRenderTarget>&  input_render_targets() { return m_input_rts; }
    inline std::vector<OutputRenderTarget>& output_render_targets() { return m_output_rts; }
    inline std::vector<InputBuffer>&        input_buffers() { return m_input_buffers; }
    inline std::vector<OutputBuffer>&       output_buffer() { return m_output_buffers; }
    inline uint32_t                         output_render_target_count() { return (uint32_t)m_output_rts.size(); }
    inline std::shared_ptr<RenderTarget>    output_render_target(const uint32_t& idx) { return m_output_rts[idx].render_target; }
    inline uint32_t                         intermediate_render_target_count() { return (uint32_t)m_intermediate_rts.size(); }
    inline std::shared_ptr<RenderTarget>    intermediate_render_target(const uint32_t& idx) { return m_intermediate_rts[idx].second; }
    inline uint32_t                         input_render_target_count() { return (uint32_t)m_input_rts.size(); }
    inline std::shared_ptr<RenderTarget>    input_render_target(const uint32_t& idx) { return m_input_rts[idx].prev_render_target; }
    inline bool                             is_enabled() { return m_enabled; }

    // Inline setters
    inline void enable() { m_enabled = true; }
    inline void disable() { m_enabled = false; }

    // Virtual methods
    virtual void        declare_connections();
    virtual bool        initialize(Renderer* renderer, ResourceManager* res_mgr) = 0;
    virtual void        execute(double delta, Renderer* renderer, Scene* scene, View* view)    = 0;
    virtual void        shutdown()                                               = 0;
    virtual std::string name()                                                   = 0;

    // Event callbacks
    virtual void on_window_resized(const uint32_t& w, const uint32_t& h);

protected:
    void                          trigger_cascade_view_render(View* view);
    void                          register_input_render_target(const std::string& name);
    void                          register_input_buffer(const std::string& name);
    std::shared_ptr<RenderTarget> register_output_render_target(const std::string& name, const uint32_t& w, const uint32_t& h, GLenum target, GLenum internal_format, GLenum format, GLenum type, uint32_t num_samples = 1, uint32_t array_size = 1, uint32_t mip_levels = 1);
    std::shared_ptr<RenderTarget> register_forwarded_output_render_target(const std::string& input);
    std::shared_ptr<RenderTarget> register_scaled_output_render_target(const std::string& name, const float& w, const float& h, GLenum target, GLenum internal_format, GLenum format, GLenum type, uint32_t num_samples = 1, uint32_t array_size = 1, uint32_t mip_levels = 1);
    std::shared_ptr<RenderTarget> register_intermediate_render_target(const std::string& name, const uint32_t& w, const uint32_t& h, GLenum target, GLenum internal_format, GLenum format, GLenum type, uint32_t num_samples = 1, uint32_t array_size = 1, uint32_t mip_levels = 1);
    std::shared_ptr<RenderTarget> register_scaled_intermediate_render_target(const std::string& name, const float& w, const float& h, GLenum target, GLenum internal_format, GLenum format, GLenum type, uint32_t num_samples = 1, uint32_t array_size = 1, uint32_t mip_levels = 1);
    void                          register_bool_parameter(const std::string& name, bool& parameter);
    void                          register_int_parameter(const std::string& name, int32_t& parameter);
    void                          register_float_parameter(const std::string& name, float& parameter);
    void                          bind_shadow_maps(Renderer* renderer, Program* program, int32_t tex_unit, uint32_t flags);

    // Geometry render helpers
    void render_scene(Renderer* renderer, Scene* scene, View* view, ShaderLibrary* library, uint32_t flags = 0, std::function<void(View*, Program*, int32_t&)> function = nullptr);
    void render_fullscreen_triangle(Renderer* renderer, View* view, Program* program = nullptr, int32_t tex_unit = 0, uint32_t flags = 0);
    void render_fullscreen_quad(Renderer* renderer, View* view, Program* program = nullptr, int32_t tex_unit = 0, uint32_t flags = 0);

protected:
    RenderGraph*                m_graph;
    float                       m_total_time_cpu;
    float                       m_total_time_gpu;
    std::string                 m_passthrough_name;
    std::vector<BoolParameter>  m_bool_parameters;
    std::vector<IntParameter>   m_int_parameters;
    std::vector<FloatParameter> m_float_parameters;

private:
    bool                                                               m_enabled;
    std::vector<OutputRenderTarget>                                    m_output_rts;
    std::vector<std::pair<std::string, std::shared_ptr<RenderTarget>>> m_intermediate_rts;
    std::vector<InputRenderTarget>                                     m_input_rts;
    std::vector<OutputBuffer>                                          m_output_buffers;
    std::vector<InputBuffer>                                           m_input_buffers;
};
} // namespace nimble