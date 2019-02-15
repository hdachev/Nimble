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
        return std::make_shared<class_name>(graph);                                 \
    }

namespace nimble
{
struct View;
struct FramebufferGroup;
class RenderGraph;
class Scene;
class ShaderLibrary;

enum RenderNodeType
{
    RENDER_NODE_SCENE      = 0,
    RENDER_NODE_FULLSCREEN = 1,
    RENDER_NODE_COMPUTE    = 2
};

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
    NODE_USAGE_DEFAULT               = NODE_USAGE_PER_OBJECT_UBO | NODE_USAGE_PER_VIEW_UBO | NODE_USAGE_POINT_LIGHTS | NODE_USAGE_SPOT_LIGHTS | NODE_USAGE_DIRECTIONAL_LIGHTS | NODE_USAGE_STATIC_MESH | NODE_USAGE_SKELETAL_MESH | NODE_USAGE_ALL_MATERIALS | NODE_USAGE_SHADOW_MAPPING
};

struct SceneRenderDesc
{
    FramebufferGroup* fbg;
    uint32_t          target_slice;
    uint32_t          x;
    uint32_t          y;
    uint32_t          w;
    uint32_t          h;
    GLenum            clear_flags      = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT;
    uint32_t          num_clear_colors = 0;
    float             clear_colors[8][4];
    double            clear_depth = 1;
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

    RenderNode(RenderNodeType type, RenderGraph* graph);
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

    // Inline getters
    inline void                             push_define(const std::string& define) { m_defines.push_back(define); }
    inline std::vector<std::string>         defines() { return m_defines; }
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
    inline RenderNodeType                   type() { return m_render_node_type; }
    inline bool                             is_enabled() { return m_enabled; }

    // Inline setters
    inline void enable() { m_enabled = true; }
    inline void disable() { m_enabled = false; }

    // Virtual methods
    virtual bool        initialize_internal();
    virtual bool        register_resources();
    virtual void        passthrough();
    virtual uint32_t    flags();
    virtual void        execute(const View* view) = 0;
    virtual bool        initialize()              = 0;
    virtual void        shutdown()                = 0;
    virtual std::string name()                    = 0;

    // Event callbacks
    virtual void on_window_resized(const uint32_t& w, const uint32_t& h);

protected:
	void						  trigger_directional_light_update(const View* view);
    void                          register_input_render_target(const std::string& name);
    void                          register_input_buffer(const std::string& name);
    std::shared_ptr<RenderTarget> register_output_render_target(const std::string& name, const uint32_t& w, const uint32_t& h, GLenum target, GLenum internal_format, GLenum format, GLenum type, uint32_t num_samples = 1, uint32_t array_size = 1, uint32_t mip_levels = 1);
    std::shared_ptr<RenderTarget> register_scaled_output_render_target(const std::string& name, const float& w, const float& h, GLenum target, GLenum internal_format, GLenum format, GLenum type, uint32_t num_samples = 1, uint32_t array_size = 1, uint32_t mip_levels = 1);
    std::shared_ptr<RenderTarget> register_intermediate_render_target(const std::string& name, const uint32_t& w, const uint32_t& h, GLenum target, GLenum internal_format, GLenum format, GLenum type, uint32_t num_samples = 1, uint32_t array_size = 1, uint32_t mip_levels = 1);
    std::shared_ptr<RenderTarget> register_scaled_intermediate_render_target(const std::string& name, const float& w, const float& h, GLenum target, GLenum internal_format, GLenum format, GLenum type, uint32_t num_samples = 1, uint32_t array_size = 1, uint32_t mip_levels = 1);

protected:
    RenderGraph* m_graph;
    float        m_total_time_cpu;
    float        m_total_time_gpu;
    std::string  m_passthrough_name;

private:
    bool                                                               m_enabled;
    RenderNodeType                                                     m_render_node_type;
    std::vector<OutputRenderTarget>                                    m_output_rts;
    std::vector<std::pair<std::string, std::shared_ptr<RenderTarget>>> m_intermediate_rts;
    std::vector<InputRenderTarget>                                     m_input_rts;
    std::vector<OutputBuffer>                                          m_output_buffers;
    std::vector<InputBuffer>                                           m_input_buffers;
    std::vector<std::string>                                           m_defines;
};

class SceneRenderNode : public RenderNode
{
public:
    struct Params
    {
        const View*       view;
        uint32_t          num_rt_views;
        RenderTargetView* rt_views;
        RenderTargetView* depth_views;
        uint32_t          x;
        uint32_t          y;
        uint32_t          w;
        uint32_t          h;
        GLenum            clear_flags      = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT;
        GLenum            cull_face        = GL_BACK;
        bool              enable_depth     = true;
        uint32_t          num_clear_colors = 0;
        float             clear_colors[8][4];
        double            clear_depth = 1;

        Params();
    };

    SceneRenderNode(RenderGraph* graph);
    ~SceneRenderNode();

    bool                initialize_internal() override;
    void                execute(const View* view) override;
    virtual std::string vs_template_path() = 0;
    virtual std::string fs_template_path() = 0;

protected:
    virtual void execute_internal(const View* view) = 0;
    virtual void set_shader_uniforms(const View* view, Program* program, int32_t& tex_unit);
    void         render_scene(const Params& params);

private:
    std::shared_ptr<ShaderLibrary> m_library;
};

class MultiPassRenderNode : public RenderNode
{
public:
    MultiPassRenderNode(RenderNodeType type, RenderGraph* graph);
    ~MultiPassRenderNode();

    void execute(const View* view) override;
    void timing_sub_pass(const uint32_t& index, std::string& name, float& cpu_time, float& gpu_time);

    // Inline getters
    inline uint32_t sub_pass_count() { return (uint32_t)m_sub_passes.size(); }

protected:
    void attach_sub_pass(const std::string& node_name, std::function<void(void)> function);

private:
    std::vector<std::pair<std::string, std::function<void(void)>>> m_sub_passes;
    std::vector<std::pair<float, float>>                           m_sub_pass_timings;
};

class FullscreenRenderNode : public MultiPassRenderNode
{
public:
    struct Params
    {
        Scene*            scene;
        View*             view;
        uint32_t          num_rt_views;
        RenderTargetView* rt_views;
        uint32_t          x;
        uint32_t          y;
        uint32_t          w;
        uint32_t          h;
        GLenum            clear_flags      = GL_COLOR_BUFFER_BIT;
        uint32_t          num_clear_colors = 0;
        float             clear_colors[8][4];

        Params();
    };

    FullscreenRenderNode(RenderGraph* graph);
    ~FullscreenRenderNode();

protected:
    void render_triangle(const Params& params);
};

class ComputeRenderNode : public MultiPassRenderNode
{
public:
    ComputeRenderNode(RenderGraph* graph);
    ~ComputeRenderNode();

    Buffer* find_output_buffer(const std::string& name);
    Buffer* find_intermediate_buffer(const std::string& name);

protected:
    std::unordered_map<std::string, Buffer*> m_output_buffers;
    std::unordered_map<std::string, Buffer*> m_intermediate_buffers;
};

} // namespace nimble