#pragma once

#include <glm.hpp>
#include <unordered_map>
#include <memory>
#include <array>
#include "ogl.h"
#include "macros.h"
#include "scene.h"
#include "view.h"
#include "uniforms.h"
#include "render_target.h"
#include "static_hash_map.h"
#include "shader_cache.h"

namespace nimble
{
class ShadowRenderGraph;
class ResourceManager;

enum ShadowMapQuality : uint32_t
{
    SHADOW_MAP_QUALITY_LOW,
    SHADOW_MAP_QUALITY_MEIDUM,
    SHADOW_MAP_QUALITY_HIGH,
    SHADOW_MAP_QUALITY_VERY_HIGH
};

class Renderer
{
public:
    struct Settings
    {
        ShadowMapQuality shadow_map_quality  = SHADOW_MAP_QUALITY_HIGH;
        uint32_t         cascade_count       = 4;
        uint32_t         sample_count        = 1;
        bool             per_cascade_culling = true;
        bool             pssm                = false;
        float            csm_lambda          = 0.5f;
    };

    Renderer(Settings settings = Settings());
    ~Renderer();

    bool initialize(ResourceManager* res_mgr, const uint32_t& w, const uint32_t& h);
    void render(double delta);
    void shutdown();

    void  set_settings(Settings settings);
    void  set_scene(std::shared_ptr<Scene> scene);
    void  register_render_graph(std::shared_ptr<RenderGraph> graph);
    void  set_scene_render_graph(std::shared_ptr<RenderGraph> graph);
    void  set_directional_light_render_graph(std::shared_ptr<ShadowRenderGraph> graph);
    void  set_spot_light_render_graph(std::shared_ptr<ShadowRenderGraph> graph);
    void  set_point_light_render_graph(std::shared_ptr<ShadowRenderGraph> graph);
    View* allocate_view();
    void  queue_view(View* view);
    void  queue_directional_light_views(View* dependent_view);
    void  queue_spot_light_views();
    void  queue_point_light_views();
    void  clear_all_views();
    void  on_window_resized(const uint32_t& w, const uint32_t& h);

    // Shader program caching.
    std::shared_ptr<Program> create_program(const std::shared_ptr<Shader>& vs, const std::shared_ptr<Shader>& fs);
    std::shared_ptr<Program> create_program(const std::vector<std::shared_ptr<Shader>>& shaders);

    Framebuffer* framebuffer_for_render_targets(const uint32_t& num_render_targets, const RenderTargetView* rt_views, const RenderTargetView* depth_view);
    void         bind_render_targets(const uint32_t& num_render_targets, const RenderTargetView* rt_views, const RenderTargetView* depth_view);

    // Inline getters
    inline ShaderCache&                       shader_cache() { return m_shader_cache; }
    inline std::weak_ptr<Scene>               scene() { return m_scene; }
    inline Settings                           settings() { return m_settings; }
    inline std::shared_ptr<RenderGraph>       scene_render_graph() { return m_scene_render_graph; }
    inline std::shared_ptr<ShadowRenderGraph> directional_light_render_graph() { return m_directional_light_render_graph; }
    inline std::shared_ptr<ShadowRenderGraph> spot_light_render_graph() { return m_spot_light_render_graph; }
    inline std::shared_ptr<ShadowRenderGraph> point_light_render_graph() { return m_point_light_render_graph; }
    inline std::shared_ptr<Texture>           directional_light_shadow_maps() { return m_directional_light_shadow_maps; }
    inline std::shared_ptr<Texture>           spot_light_shadow_maps() { return m_spot_light_shadow_maps; }
    inline std::shared_ptr<Texture>           point_light_shadow_maps() { return m_point_light_shadow_maps; }
    inline ShaderStorageBuffer*               per_view_ssbo() { return m_per_view.get(); }
    inline UniformBuffer*                     per_entity_ubo() { return m_per_entity.get(); }
    inline ShaderStorageBuffer*               per_scene_ssbo() { return m_per_scene.get(); }
    inline std::shared_ptr<VertexArray>       cube_vao() { return m_cube_vao; }

private:
    using TextureLifetimes = std::vector<std::pair<uint32_t, uint32_t>>;

    struct RenderTargetDesc
    {
        std::shared_ptr<RenderTarget> rt;
        TextureLifetimes              lifetimes;
    };

    void     setup_cascade_views(DirectionalLight& dir_light, View* dependent_view, View** cascade_views, View* parent = nullptr);
    void     create_cube();
    int32_t  find_render_target_last_usage(std::shared_ptr<RenderTarget> rt);
    bool     is_aliasing_candidate(std::shared_ptr<RenderTarget> rt, uint32_t write_node, uint32_t read_node, const RenderTargetDesc& rt_desc);
    void     create_texture_for_render_target(std::shared_ptr<RenderTarget> rt, uint32_t write_node, uint32_t read_node);
    void     bake_render_graphs();
    void     update_uniforms();
    void     cull_scene();
    bool     queue_rendered_view(View* view);
    uint32_t queue_update_view(View* view);
    uint32_t queue_culled_view(Frustum f);
    void     queue_default_views();
    void     render_all_views(double delta);

private:
    // Resource caches
    ShaderCache                                             m_shader_cache;
    StaticHashMap<uint64_t, Framebuffer*, 1024>             m_fbo_cache;
    std::unordered_map<std::string, std::weak_ptr<Program>> m_program_cache;
    std::vector<RenderTargetDesc>                           m_rt_cache;
    uint32_t                                                m_window_width;
    uint32_t                                                m_window_height;

    // Current scene.
    uint32_t                                    m_num_cull_views      = 0;
    uint32_t                                    m_num_update_views    = 0;
    uint32_t                                    m_num_rendered_views  = 0;
    uint32_t                                    m_num_allocated_views = 0;
    std::array<View, MAX_VIEWS>                 m_view_pool;
    std::array<View*, MAX_VIEWS>                m_update_views;
    std::array<View*, MAX_VIEWS>                m_rendered_views;
    std::array<Frustum, MAX_VIEWS>              m_active_frustums;
    std::weak_ptr<Scene>                        m_scene;
    std::shared_ptr<RenderGraph>                m_scene_render_graph             = nullptr;
    std::shared_ptr<ShadowRenderGraph>          m_directional_light_render_graph = nullptr;
    std::shared_ptr<ShadowRenderGraph>          m_spot_light_render_graph        = nullptr;
    std::shared_ptr<ShadowRenderGraph>          m_point_light_render_graph       = nullptr;
    std::vector<std::shared_ptr<RenderGraph>>   m_registered_render_graphs;
    std::array<PerViewUniforms, MAX_VIEWS>      m_per_view_uniforms;
    std::array<PerEntityUniforms, MAX_ENTITIES> m_per_entity_uniforms;
    PerSceneUniforms                            m_per_scene_uniforms;

    // Uniform buffers
    std::unique_ptr<ShaderStorageBuffer> m_per_view;
    std::unique_ptr<UniformBuffer>       m_per_entity;
    std::unique_ptr<ShaderStorageBuffer> m_per_scene;

    // Shadow Maps
    std::shared_ptr<Texture>      m_directional_light_shadow_maps;
    std::shared_ptr<Texture>      m_spot_light_shadow_maps;
    std::shared_ptr<Texture>      m_point_light_shadow_maps;
    std::vector<RenderTargetView> m_directionl_light_rt_views;
    std::vector<RenderTargetView> m_point_light_rt_views;
    std::vector<RenderTargetView> m_spot_light_rt_views;
    Settings                      m_settings;

    // Common geometry.
    std::shared_ptr<VertexArray>  m_cube_vao;
    std::shared_ptr<VertexBuffer> m_cube_vbo;
};
} // namespace nimble