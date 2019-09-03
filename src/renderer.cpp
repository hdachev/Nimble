#include "renderer.h"
#include "camera.h"
#include "material.h"
#include "mesh.h"
#include "logger.h"
#include "utility.h"
#include "imgui.h"
#include "entity.h"
#include "constants.h"
#include "profiler.h"
#include "render_graph.h"
#include "geometry.h"
#include "profiler.h"
#include "resource_manager.h"
#include "global_probe_renderer.h"
#include "local_probe_renderer.h"
#include "viewport_manager.h"

#include <GLFW/glfw3.h>
#include <gtc/matrix_transform.hpp>
#include <fstream>

namespace nimble
{
#define POINT_LIGHT_NEAR_PLANE 0.1f

static const uint32_t kDirectionalLightShadowMapSizes[] = {
    512,
    1024,
    2048,
    4096
};

static const uint32_t kSpotLightShadowMapSizes[] = {
    256,
    512,
    1024,
    2048
};

static const uint32_t kPointShadowMapSizes[] = {
    128,
    256,
    512,
    1024
};

struct FrustumSplit
{
    float     near_plane;
    float     far_plane;
    float     ratio;
    float     fov;
    glm::vec3 center;
    glm::vec3 corners[8];
};

struct RenderTargetKey
{
    uint32_t face      = UINT32_MAX;
    uint32_t layer     = UINT32_MAX;
    uint32_t mip_level = UINT32_MAX;
    uint32_t gl_id     = UINT32_MAX;
    uint32_t version   = UINT32_MAX;
};

struct FramebufferKey
{
    RenderTargetKey rt_keys[8];
    RenderTargetKey depth_key;
};

static glm::vec3 s_cube_view_params[6][2] = {
    { glm::vec3(1.0, 0.0, 0.0), glm::vec3(0.0, -1.0, 0.0) },
    { glm::vec3(-1.0, 0.0, 0.0), glm::vec3(0.0, -1.0, 0.0) },
    { glm::vec3(0.0, 1.0, 0.0), glm::vec3(0.0, 0.0, 1.0) },
    { glm::vec3(0.0, -1.0, 0.0), glm::vec3(0.0, 0.0, -1.0) },
    { glm::vec3(0.0, 0.0, 1.0), glm::vec3(0.0, -1.0, 0.0) },
    { glm::vec3(0.0, 0.0, -1.0), glm::vec3(0.0, -1.0, 0.0) }
};

// -----------------------------------------------------------------------------------------------------------------------------------

Renderer::Renderer(Settings settings) :
    m_settings(settings) {}

// -----------------------------------------------------------------------------------------------------------------------------------

Renderer::~Renderer() {}

// -----------------------------------------------------------------------------------------------------------------------------------

bool Renderer::initialize(ResourceManager* res_mgr, const uint32_t& w, const uint32_t& h)
{
    m_window_width  = w;
    m_window_height = h;

    m_directional_light_shadow_map_depth_attachment.reset();
    m_spot_light_shadow_map_depth_attachment.reset();
    m_point_light_shadow_map_depth_attachment.reset();

    // Create shadow maps
    m_directional_light_shadow_map_depth_attachment = std::make_shared<Texture2D>(kDirectionalLightShadowMapSizes[m_settings.shadow_map_quality], kDirectionalLightShadowMapSizes[m_settings.shadow_map_quality], m_settings.cascade_count * MAX_SHADOW_CASTING_DIRECTIONAL_LIGHTS, 1, 1, GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT, false);
    m_spot_light_shadow_map_depth_attachment        = std::make_shared<Texture2D>(kSpotLightShadowMapSizes[m_settings.shadow_map_quality], kSpotLightShadowMapSizes[m_settings.shadow_map_quality], MAX_SHADOW_CASTING_SPOT_LIGHTS, 1, 1, GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT, false);
    m_point_light_shadow_map_depth_attachment       = std::make_shared<TextureCube>(kPointShadowMapSizes[m_settings.shadow_map_quality], kPointShadowMapSizes[m_settings.shadow_map_quality], MAX_SHADOW_CASTING_POINT_LIGHTS, 1, GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT, false);

    m_directional_light_shadow_map_depth_attachment->set_min_filter(GL_LINEAR);
    m_directional_light_shadow_map_depth_attachment->set_mag_filter(GL_LINEAR);
    m_directional_light_shadow_map_depth_attachment->set_wrapping(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);
    m_directional_light_shadow_map_depth_attachment->set_compare_mode(GL_COMPARE_REF_TO_TEXTURE);
    m_directional_light_shadow_map_depth_attachment->set_compare_func(GL_LEQUAL);

    m_spot_light_shadow_map_depth_attachment->set_min_filter(GL_LINEAR);
    m_spot_light_shadow_map_depth_attachment->set_mag_filter(GL_LINEAR);
    m_spot_light_shadow_map_depth_attachment->set_compare_mode(GL_COMPARE_REF_TO_TEXTURE);
    m_spot_light_shadow_map_depth_attachment->set_compare_func(GL_LEQUAL);

    m_point_light_shadow_map_depth_attachment->set_min_filter(GL_LINEAR);
    m_point_light_shadow_map_depth_attachment->set_mag_filter(GL_LINEAR);
    m_point_light_shadow_map_depth_attachment->set_compare_mode(GL_COMPARE_REF_TO_TEXTURE);
    m_point_light_shadow_map_depth_attachment->set_compare_func(GL_LEQUAL);

    // Create shadow map Render Target Views
    for (uint32_t i = 0; i < MAX_SHADOW_CASTING_DIRECTIONAL_LIGHTS; i++)
    {
        for (uint32_t j = 0; j < m_settings.cascade_count; j++)
            m_directionl_light_depth_rt_views.push_back({ 0, i * m_settings.cascade_count + j, 0, m_directional_light_shadow_map_depth_attachment });
    }

    for (uint32_t i = 0; i < MAX_SHADOW_CASTING_SPOT_LIGHTS; i++)
        m_spot_light_depth_rt_views.push_back({ 0, i, 0, m_spot_light_shadow_map_depth_attachment });

    for (uint32_t i = 0; i < MAX_SHADOW_CASTING_POINT_LIGHTS; i++)
    {
        for (uint32_t j = 0; j < 6; j++)
            m_point_light_depth_rt_views.push_back({ j, i, 0, m_point_light_shadow_map_depth_attachment });
    }

    // Common resources
    m_per_view   = std::make_unique<ShaderStorageBuffer>(GL_DYNAMIC_DRAW, MAX_VIEWS * sizeof(PerViewUniforms));
    m_per_entity = std::make_unique<UniformBuffer>(GL_DYNAMIC_DRAW, MAX_ENTITIES * sizeof(PerEntityUniforms));
    m_per_scene  = std::make_unique<ShaderStorageBuffer>(GL_DYNAMIC_DRAW, sizeof(PerSceneUniforms));

    create_cube();

    bake_render_graphs();

    for (auto& current_graph : m_registered_render_graphs)
    {
        current_graph->on_window_resized(m_window_width, m_window_height);

        if (!current_graph->initialize(this, res_mgr))
            return false;
    }

    if (m_global_probe_renderer)
        m_global_probe_renderer->initialize(this, res_mgr);

    m_debug_vs = res_mgr->load_shader("shader/post_process/fullscreen_triangle_vs.glsl", GL_VERTEX_SHADER);
    m_debug_fs = res_mgr->load_shader("shader/post_process/debug_fs.glsl", GL_FRAGMENT_SHADER);

    if (m_debug_vs && m_debug_fs)
        m_debug_program = create_program(m_debug_vs, m_debug_fs);
    else
        return false;

    m_copy_vs = res_mgr->load_shader("shader/post_process/fullscreen_triangle_vs.glsl", GL_VERTEX_SHADER);
    m_copy_fs = res_mgr->load_shader("shader/post_process/copy_fs.glsl", GL_FRAGMENT_SHADER);

    if (m_copy_vs && m_debug_fs)
        m_copy_program = create_program(m_copy_vs, m_copy_fs);
    else
        return false;

    return true;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Renderer::render(double delta, ViewportManager* viewport_mgr)
{
    render_probes(delta);

    queue_default_views();

    update_uniforms(delta);

    cull_scene();

    render_all_views(delta);

    viewport_mgr->render_viewports(this, m_num_rendered_views, &m_rendered_views[0]);

    clear_all_views();

    render_debug_output();
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Renderer::shutdown()
{
    // Delete common geometry VBO's and VAO's.
    m_cube_vao.reset();
    m_cube_vbo.reset();

    // Delete framebuffer
    for (uint32_t i = 0; i < m_fbo_cache.size(); i++)
    {
        NIMBLE_SAFE_DELETE(m_fbo_cache.m_value[i]);
        m_fbo_cache.remove(m_fbo_cache.m_key[i]);
    }

    // Clean up Shader Cache
    m_shader_cache.shutdown();

    // Delete programs.
    for (auto itr : m_program_cache)
        itr.second.reset();

    m_per_view.reset();
    m_per_entity.reset();
    m_per_scene.reset();

    m_directional_light_shadow_map_depth_attachment.reset();
    m_spot_light_shadow_map_depth_attachment.reset();
    m_point_light_shadow_map_depth_attachment.reset();
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Renderer::set_settings(Settings settings)
{
    m_settings = settings;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Renderer::set_scene(std::shared_ptr<Scene> scene)
{
    m_scene = scene;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Renderer::register_render_graph(std::shared_ptr<RenderGraph> graph)
{
    for (auto& current_graph : m_registered_render_graphs)
    {
        if (current_graph->name() == graph->name())
        {
            NIMBLE_LOG_WARNING("Attempting to register the same Render Graph twice: " + graph->name());
            return;
        }
    }

    m_registered_render_graphs.push_back(graph);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Renderer::set_scene_render_graph(std::shared_ptr<RenderGraph> graph)
{
    if (graph)
        m_scene_render_graph = graph;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Renderer::set_directional_light_render_graph(std::shared_ptr<ShadowRenderGraph> graph)
{
    if (graph)
        m_directional_light_render_graph = graph;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Renderer::set_spot_light_render_graph(std::shared_ptr<ShadowRenderGraph> graph)
{
    if (graph)
        m_spot_light_render_graph = graph;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Renderer::set_point_light_render_graph(std::shared_ptr<ShadowRenderGraph> graph)
{
    if (graph)
        m_point_light_render_graph = graph;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Renderer::set_global_probe_renderer(std::shared_ptr<GlobalProbeRenderer> probe_renderer)
{
    m_global_probe_renderer = probe_renderer;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Renderer::set_local_probe_renderer(std::shared_ptr<LocalProbeRenderer> probe_renderer)
{
    m_local_probe_renderer = probe_renderer;
}

// -----------------------------------------------------------------------------------------------------------------------------------

View* Renderer::allocate_view()
{
    if (m_num_allocated_views == MAX_VIEWS)
    {
        NIMBLE_LOG_ERROR("Maximum number of Views reached (64)");
        return nullptr;
    }
    else
    {
        uint32_t idx = m_num_allocated_views++;

        View* view = &m_view_pool[idx];

        view->dest_depth_render_target_view = nullptr;
        view->num_cascade_frustums          = 0;
        view->num_cascade_views             = 0;

        return view;
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Renderer::queue_view(View* view)
{
    Frustum f;
    frustum_from_matrix(f, view->vp_mat);

    uint32_t cull_idx    = queue_culled_view(f);
    uint32_t uniform_idx = queue_update_view(view);

    if (cull_idx != UINT32_MAX && uniform_idx != UINT32_MAX)
    {
        view->cull_idx    = cull_idx;
        view->uniform_idx = uniform_idx;
        queue_rendered_view(view);
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Renderer::queue_directional_light_views(View* dependent_view)
{
    if (!m_scene.expired())
    {
        auto scene = m_scene.lock();

        uint32_t          shadow_casting_light_idx = 0;
        uint32_t          i                        = 0;
        DirectionalLight* lights                   = scene->directional_lights();
        dependent_view->num_cascade_frustums       = 0;

        for (uint32_t light_idx = 0; light_idx < scene->directional_light_count(); light_idx++)
        {
            DirectionalLight& light = lights[light_idx];

            if (light.casts_shadow)
            {
                View*    cascade_views[MAX_SHADOW_MAP_CASCADES];
                View     temp;
                View*    parent = nullptr;
                uint32_t cull_idx;

                // Allocate Views for shadow cascades and fill out initial values
                for (uint32_t cascade_idx = 0; cascade_idx < m_settings.cascade_count; cascade_idx++)
                {
                    View* light_view = allocate_view();

                    light_view->tag                                 = "Directional Light " + std::to_string(light_idx) + " Cascade View " + std::to_string(cascade_idx);
                    light_view->enabled                             = true;
                    light_view->culling                             = true;
                    light_view->direction                           = light.transform.forward();
                    light_view->position                            = light.transform.position;
                    light_view->prev_vp_mat                         = glm::mat4(1.0f);
                    light_view->inv_view_mat                        = glm::mat4(1.0f);
                    light_view->inv_projection_mat                  = glm::mat4(1.0f);
                    light_view->inv_vp_mat                          = glm::mat4(1.0f);
                    light_view->jitter                              = glm::vec4(0.0);
                    light_view->dest_color_render_target_view_count = 0;
                    light_view->dest_color_render_target_views      = nullptr;
                    light_view->dest_depth_render_target_view       = &m_directionl_light_depth_rt_views[shadow_casting_light_idx * m_settings.cascade_count + cascade_idx];
                    light_view->graph                               = m_directional_light_render_graph;
                    light_view->type                                = VIEW_DIRECTIONAL_LIGHT;
                    light_view->light_index                         = light_idx;

                    cascade_views[cascade_idx] = light_view;

                    dependent_view->num_cascade_frustums++;
                }

                // If per cascade culling is disabled, assign parent View
                if (!dependent_view->graph->per_cascade_culling())
                    parent = &temp;

                // Calculate cascade matrices
                setup_cascade_views(light, dependent_view, cascade_views, parent);

                // If per cascade culling is disabled, queue up a culling View for the parent View
                if (!dependent_view->graph->per_cascade_culling())
                {
                    Frustum f;
                    frustum_from_matrix(f, parent->vp_mat);
                    cull_idx = queue_culled_view(f);
                }

                for (uint32_t cascade_idx = 0; cascade_idx < m_settings.cascade_count; cascade_idx++)
                {
                    // If per cascade culling is enabled, queue up culling views for each cascade
                    if (dependent_view->graph->per_cascade_culling())
                    {
                        Frustum f;
                        frustum_from_matrix(f, cascade_views[cascade_idx]->vp_mat);
                        cull_idx = queue_culled_view(f);
                    }

                    cascade_views[cascade_idx]->cull_idx = cull_idx;

                    uint32_t uniform_idx                    = queue_update_view(cascade_views[cascade_idx]);
                    cascade_views[cascade_idx]->uniform_idx = uniform_idx;

                    if (dependent_view->graph->is_manual_cascade_rendering())
                        dependent_view->cascade_views[i++] = cascade_views[cascade_idx];
                    else
                        queue_rendered_view(cascade_views[cascade_idx]);
                }

                shadow_casting_light_idx++;
            }

            // Stop adding views if max number of shadow casting lights is already queued.
            if (shadow_casting_light_idx == (MAX_SHADOW_CASTING_DIRECTIONAL_LIGHTS - 1))
                break;
        }

        if (dependent_view->graph->is_manual_cascade_rendering())
            dependent_view->num_cascade_views = i;
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Renderer::queue_spot_light_views()
{
    if (!m_scene.expired())
    {
        auto scene = m_scene.lock();

        uint32_t   shadow_casting_light_idx = 0;
        SpotLight* lights                   = scene->spot_lights();

        for (uint32_t light_idx = 0; light_idx < scene->spot_light_count(); light_idx++)
        {
            SpotLight& light = lights[light_idx];

            if (light.casts_shadow)
            {
                View* light_view = allocate_view();

                light_view->tag                                 = "Spot Light View " + std::to_string(light_idx);
                light_view->enabled                             = true;
                light_view->culling                             = true;
                light_view->direction                           = light.transform.forward();
                light_view->position                            = light.transform.position;
                light_view->view_mat                            = glm::lookAt(light_view->position, light_view->position + light_view->direction, glm::vec3(0.0f, 1.0f, 0.0f));
                light_view->projection_mat                      = glm::perspective(glm::radians(2.0f * light.outer_cone_angle), 1.0f, 1.0f, light.range);
                light_view->vp_mat                              = light_view->projection_mat * light_view->view_mat;
                light_view->prev_vp_mat                         = glm::mat4(1.0f);
                light_view->inv_view_mat                        = glm::mat4(1.0f);
                light_view->inv_projection_mat                  = glm::mat4(1.0f);
                light_view->inv_vp_mat                          = glm::mat4(1.0f);
                light_view->jitter                              = glm::vec4(0.0);
                light_view->dest_color_render_target_view_count = 0;
                light_view->dest_color_render_target_views      = nullptr;
                light_view->dest_depth_render_target_view       = &m_spot_light_depth_rt_views[shadow_casting_light_idx];
                light_view->graph                               = m_spot_light_render_graph;
                light_view->type                                = VIEW_SPOT_LIGHT;
                light_view->light_index                         = light_idx;

                m_per_scene_uniforms.spot_light_shadow_matrix[shadow_casting_light_idx] = light_view->vp_mat;

                queue_view(light_view);

                shadow_casting_light_idx++;
            }

            // Stop adding views if max number of shadow casting lights is already queued.
            if (shadow_casting_light_idx == (MAX_SHADOW_CASTING_SPOT_LIGHTS - 1))
                break;
        }
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Renderer::queue_point_light_views()
{
    if (!m_scene.expired())
    {
        auto scene = m_scene.lock();

        uint32_t    shadow_casting_light_idx = 0;
        PointLight* lights                   = scene->point_lights();

        for (uint32_t light_idx = 0; light_idx < scene->point_light_count(); light_idx++)
        {
            PointLight& light = lights[light_idx];

            if (light.casts_shadow)
            {
                for (uint32_t face_idx = 0; face_idx < 6; face_idx++)
                {
                    View* light_view = allocate_view();

                    light_view->tag                                 = "Point Light View " + std::to_string(light_idx) + " - " + std::to_string(face_idx);
                    light_view->enabled                             = true;
                    light_view->culling                             = true;
                    light_view->direction                           = light.transform.forward();
                    light_view->position                            = light.transform.position;
                    light_view->view_mat                            = glm::lookAt(light.transform.position, light.transform.position + s_cube_view_params[face_idx][0], s_cube_view_params[face_idx][1]);
                    light_view->projection_mat                      = glm::perspective(glm::radians(90.0f), 1.0f, POINT_LIGHT_NEAR_PLANE, light.range);
                    light_view->vp_mat                              = light_view->projection_mat * light_view->view_mat;
                    light_view->prev_vp_mat                         = glm::mat4(1.0f);
                    light_view->inv_view_mat                        = glm::inverse(light_view->view_mat);
                    light_view->inv_projection_mat                  = glm::inverse(light_view->projection_mat);
                    light_view->inv_vp_mat                          = glm::inverse(light_view->vp_mat);
                    light_view->jitter                              = glm::vec4(0.0);
                    light_view->dest_color_render_target_view_count = 0;
                    light_view->dest_color_render_target_views      = nullptr;
                    light_view->dest_depth_render_target_view       = &m_point_light_depth_rt_views[shadow_casting_light_idx * 6 + face_idx];
                    light_view->graph                               = m_point_light_render_graph;
                    light_view->type                                = VIEW_POINT_LIGHT;
                    light_view->light_index                         = light_idx;

                    queue_view(light_view);
                }

                shadow_casting_light_idx++;
            }

            // Stop adding views if max number of shadow casting lights is already queued.
            if (shadow_casting_light_idx == (MAX_SHADOW_CASTING_POINT_LIGHTS - 1))
                break;
        }
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Renderer::clear_all_views()
{
    m_num_cull_views      = 0;
    m_num_update_views    = 0;
    m_num_rendered_views  = 0;
    m_num_allocated_views = 0;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Renderer::on_window_resized(const uint32_t& w, const uint32_t& h)
{
    m_window_width  = w;
    m_window_height = h;

    for (auto& desc : m_rt_cache)
    {
        if (desc.rt->is_scaled() && desc.rt->target == GL_TEXTURE_2D)
        {
            uint32_t width  = uint32_t(float(w) * desc.rt->scale_w);
            uint32_t height = uint32_t(float(h) * desc.rt->scale_h);

            Texture2D* texture = (Texture2D*)desc.rt->texture.get();
            texture->resize(width, height);
            texture->set_wrapping(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);
        }
    }

    if (m_scene_render_graph)
        m_scene_render_graph->on_window_resized(w, h);
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::shared_ptr<Program> Renderer::create_program(const std::shared_ptr<Shader>& vs, const std::shared_ptr<Shader>& fs)
{
    std::string id = std::to_string(vs->id()) + "-";
    id += std::to_string(fs->id());

    if (m_program_cache.find(id) != m_program_cache.end() && m_program_cache[id].lock())
        return m_program_cache[id].lock();
    else
    {
        Shader* shaders[] = { vs.get(), fs.get() };

        std::shared_ptr<Program> program = std::make_shared<Program>(2, shaders);

        if (!program)
        {
            NIMBLE_LOG_ERROR("Program failed to link!");
            return nullptr;
        }

        m_program_cache[id] = program;

        if (program->num_active_uniform_blocks() > 0)
            program->uniform_block_binding("u_PerEntity", 1);

        return program;
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::shared_ptr<Program> Renderer::create_program(const std::vector<std::shared_ptr<Shader>>& shaders)
{
    std::vector<Shader*> shaders_raw;
    std::string          id = "";

    for (const auto& shader : shaders)
    {
        shaders_raw.push_back(shader.get());
        id += std::to_string(shader->id());
        id += "-";
    }

    if (m_program_cache.find(id) != m_program_cache.end() && m_program_cache[id].lock())
        return m_program_cache[id].lock();
    else
    {
        std::shared_ptr<Program> program = std::make_shared<Program>(shaders_raw.size(), shaders_raw.data());

        if (!program)
        {
            NIMBLE_LOG_ERROR("Program failed to link!");
            return nullptr;
        }

        m_program_cache[id] = program;

        if (program->num_active_uniform_blocks() > 0)
        {
            program->uniform_block_binding("u_PerView", 0);
            program->uniform_block_binding("u_PerScene", 1);
            program->uniform_block_binding("u_PerEntity", 2);
        }

        return program;
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

Framebuffer* Renderer::framebuffer_for_render_targets(const uint32_t& num_render_targets, const RenderTargetView* rt_views, const RenderTargetView* depth_view)
{
    FramebufferKey key;

    if (rt_views)
    {
        for (uint32_t i = 0; i < num_render_targets; i++)
        {
            key.rt_keys[i].face      = rt_views[i].face;
            key.rt_keys[i].layer     = rt_views[i].layer;
            key.rt_keys[i].mip_level = rt_views[i].mip_level;
            key.rt_keys[i].gl_id     = rt_views[i].texture->id();
            key.rt_keys[i].version   = rt_views[i].texture->version();
        }
    }

    if (depth_view)
    {
        key.depth_key.face      = depth_view->face;
        key.depth_key.layer     = depth_view->layer;
        key.depth_key.mip_level = depth_view->mip_level;
        key.depth_key.gl_id     = depth_view->texture->id();
        key.depth_key.version   = depth_view->texture->version();
    }

    uint64_t hash = murmur_hash_64(&key, sizeof(FramebufferKey), 5234);

    Framebuffer* fbo = nullptr;

    if (!m_fbo_cache.get(hash, fbo))
    {
        fbo = new Framebuffer();

        if (rt_views)
        {
            if (num_render_targets == 1)
            {
                if (rt_views[0].texture->target() == GL_TEXTURE_2D || rt_views[0].texture->target() == GL_TEXTURE_2D_ARRAY)
                    fbo->attach_render_target(0, rt_views[0].texture.get(), rt_views[0].layer, rt_views[0].mip_level);
                else if (rt_views[0].texture->target() == GL_TEXTURE_CUBE_MAP || rt_views[0].texture->target() == GL_TEXTURE_CUBE_MAP_ARRAY)
                    fbo->attach_render_target(0, static_cast<TextureCube*>(rt_views[0].texture.get()), rt_views[0].face, rt_views[0].layer, rt_views[0].mip_level);
            }
            else
            {
                Texture* textures[8];

                for (uint32_t i = 0; i < num_render_targets; i++)
                    textures[i] = rt_views[i].texture.get();

                fbo->attach_multiple_render_targets(num_render_targets, textures);
            }
        }

        if (depth_view)
        {
            if (depth_view->texture->target() == GL_TEXTURE_2D || depth_view->texture->target() == GL_TEXTURE_2D_ARRAY)
                fbo->attach_depth_stencil_target(depth_view->texture.get(), depth_view->layer, depth_view->mip_level);
            else if (depth_view->texture->target() == GL_TEXTURE_CUBE_MAP || depth_view->texture->target() == GL_TEXTURE_CUBE_MAP_ARRAY)
                fbo->attach_depth_stencil_target(static_cast<TextureCube*>(depth_view->texture.get()), depth_view->face, depth_view->layer, depth_view->mip_level);
        }

        m_fbo_cache.set(hash, fbo);
    }

    return fbo;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Renderer::bind_render_targets(const uint32_t& num_render_targets, const RenderTargetView* rt_views, const RenderTargetView* depth_view)
{
    Framebuffer* fbo = framebuffer_for_render_targets(num_render_targets, rt_views, depth_view);

    if (fbo)
        fbo->bind();
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Renderer::create_shadow_maps()
{
    auto dir_light_node = m_directional_light_render_graph->shadow_node();

    if (dir_light_node)
    {
        m_directional_light_shadow_map_depth_attachment.reset();
        m_directional_light_shadow_map_color_attachments.clear();
        m_directionl_light_depth_rt_views.clear();
        m_directionl_light_color_rt_views.clear();

        // Create shadow maps
        m_directional_light_shadow_map_depth_attachment = std::make_shared<Texture2D>(kDirectionalLightShadowMapSizes[m_settings.shadow_map_quality], kDirectionalLightShadowMapSizes[m_settings.shadow_map_quality], m_settings.cascade_count * MAX_SHADOW_CASTING_DIRECTIONAL_LIGHTS, 1, 1, dir_light_node->shadow_map_depth_format(), GL_DEPTH_COMPONENT, GL_FLOAT, false);

        m_directional_light_shadow_map_depth_attachment->set_min_filter(GL_LINEAR);
        m_directional_light_shadow_map_depth_attachment->set_mag_filter(GL_LINEAR);
        m_directional_light_shadow_map_depth_attachment->set_wrapping(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);
        m_directional_light_shadow_map_depth_attachment->set_compare_mode(GL_COMPARE_REF_TO_TEXTURE);
        m_directional_light_shadow_map_depth_attachment->set_compare_func(GL_LEQUAL);

        auto color_formats = dir_light_node->shadow_map_color_formats();

        m_directional_light_shadow_map_color_attachments.resize(color_formats.size());

        for (int i = 0; i < color_formats.size(); i++)
            m_directional_light_shadow_map_color_attachments[i] = std::make_shared<Texture2D>(kDirectionalLightShadowMapSizes[m_settings.shadow_map_quality], kDirectionalLightShadowMapSizes[m_settings.shadow_map_quality], m_settings.cascade_count * MAX_SHADOW_CASTING_DIRECTIONAL_LIGHTS, 1, 1, color_formats[i], format_from_internal_format(color_formats[i]), type_from_internal_format(color_formats[i]), false);

        // Create shadow map Render Target Views
        for (uint32_t i = 0; i < MAX_SHADOW_CASTING_DIRECTIONAL_LIGHTS; i++)
        {
            for (uint32_t j = 0; j < m_settings.cascade_count; j++)
                m_directionl_light_depth_rt_views.push_back({ 0, i * m_settings.cascade_count + j, 0, m_directional_light_shadow_map_depth_attachment });
        }

        for (int fmt = 0; fmt < color_formats.size(); fmt++)
        {
            for (uint32_t i = 0; i < MAX_SHADOW_CASTING_DIRECTIONAL_LIGHTS; i++)
            {
                for (uint32_t j = 0; j < m_settings.cascade_count; j++)
                {
                    for (uint32_t j = 0; j < m_settings.cascade_count; j++)
                        m_directionl_light_color_rt_views[fmt].push_back({ 0, i * m_settings.cascade_count + j, 0, m_directional_light_shadow_map_color_attachments[fmt] });
                }
            }
        }
    }

    auto spot_light_node = m_spot_light_render_graph->shadow_node();

    if (spot_light_node)
    {
        m_spot_light_shadow_map_depth_attachment.reset();
        m_spot_light_shadow_map_color_attachments.clear();

        // Create shadow maps
        m_spot_light_shadow_map_depth_attachment = std::make_shared<Texture2D>(kSpotLightShadowMapSizes[m_settings.shadow_map_quality], kSpotLightShadowMapSizes[m_settings.shadow_map_quality], MAX_SHADOW_CASTING_SPOT_LIGHTS, 1, 1, GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT, false);

        m_spot_light_shadow_map_depth_attachment->set_min_filter(GL_LINEAR);
        m_spot_light_shadow_map_depth_attachment->set_mag_filter(GL_LINEAR);
        m_spot_light_shadow_map_depth_attachment->set_compare_mode(GL_COMPARE_REF_TO_TEXTURE);
        m_spot_light_shadow_map_depth_attachment->set_compare_func(GL_LEQUAL);

        auto color_formats = spot_light_node->shadow_map_color_formats();

        m_spot_light_shadow_map_color_attachments.reserve(color_formats.size());

        for (int i = 0; i < color_formats.size(); i++)
            m_spot_light_shadow_map_color_attachments[i] = std::make_shared<Texture2D>(kSpotLightShadowMapSizes[m_settings.shadow_map_quality], kSpotLightShadowMapSizes[m_settings.shadow_map_quality], MAX_SHADOW_CASTING_SPOT_LIGHTS, 1, 1, color_formats[i], format_from_internal_format(color_formats[i]), type_from_internal_format(color_formats[i]), false);

        // Create shadow map Render Target Views
        for (uint32_t i = 0; i < MAX_SHADOW_CASTING_SPOT_LIGHTS; i++)
            m_spot_light_depth_rt_views.push_back({ 0, i, 0, m_spot_light_shadow_map_depth_attachment });

        for (int fmt = 0; fmt < color_formats.size(); fmt++)
        {
            for (uint32_t i = 0; i < MAX_SHADOW_CASTING_SPOT_LIGHTS; i++)
                m_spot_light_color_rt_views[fmt].push_back({ 0, i, 0, m_spot_light_shadow_map_color_attachments[i] });
        }
    }

    auto point_light_node = m_point_light_render_graph->shadow_node();

    if (point_light_node)
    {
        m_point_light_shadow_map_depth_attachment.reset();
        m_point_light_shadow_map_color_attachments.clear();

        // Create shadow maps
        m_point_light_shadow_map_depth_attachment = std::make_shared<TextureCube>(kPointShadowMapSizes[m_settings.shadow_map_quality], kPointShadowMapSizes[m_settings.shadow_map_quality], MAX_SHADOW_CASTING_POINT_LIGHTS, 1, GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT, false);

        m_point_light_shadow_map_depth_attachment->set_min_filter(GL_LINEAR);
        m_point_light_shadow_map_depth_attachment->set_mag_filter(GL_LINEAR);
        m_point_light_shadow_map_depth_attachment->set_compare_mode(GL_COMPARE_REF_TO_TEXTURE);
        m_point_light_shadow_map_depth_attachment->set_compare_func(GL_LEQUAL);

        auto color_formats = point_light_node->shadow_map_color_formats();

        m_point_light_shadow_map_color_attachments.reserve(color_formats.size());

        for (uint32_t i = 0; i < color_formats.size(); i++)
            m_point_light_shadow_map_color_attachments[i] = std::make_shared<TextureCube>(kPointShadowMapSizes[m_settings.shadow_map_quality], kPointShadowMapSizes[m_settings.shadow_map_quality], MAX_SHADOW_CASTING_POINT_LIGHTS, 1, color_formats[i], format_from_internal_format(color_formats[i]), type_from_internal_format(color_formats[i]), false);

        // Create shadow map Render Target Views
        for (uint32_t i = 0; i < MAX_SHADOW_CASTING_POINT_LIGHTS; i++)
        {
            for (uint32_t j = 0; j < 6; j++)
                m_point_light_depth_rt_views.push_back({ j, i, 0, m_point_light_shadow_map_depth_attachment });
        }

        for (uint32_t fmt = 0; fmt < color_formats.size(); fmt++)
        {
            for (uint32_t i = 0; i < MAX_SHADOW_CASTING_POINT_LIGHTS; i++)
            {
                for (uint32_t j = 0; j < 6; j++)
                    m_point_light_color_rt_views[fmt].push_back({ j, i, 0, m_point_light_shadow_map_color_attachments[fmt] });
            }
        }
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Renderer::render_probes(double delta)
{
    if (!m_scene.expired())
    {
        auto scene = m_scene.lock();

        if (m_global_probe_renderer)
            m_global_probe_renderer->render(delta, this, scene.get());
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Renderer::setup_cascade_views(DirectionalLight& dir_light, View* dependent_view, View** cascade_views, View* parent)
{
    FrustumSplit splits[MAX_SHADOW_MAP_CASCADES];
    glm::mat4    proj_matrices[MAX_SHADOW_MAP_CASCADES];
    glm::mat4    crop[MAX_SHADOW_MAP_CASCADES];

    glm::mat4 bias = glm::mat4(0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f, 0.5f, 0.5f, 0.5f, 1.0f);

    int count = m_settings.cascade_count;

    if (parent)
        count++;

    for (int i = 0; i < count; i++)
    {
        splits[i].fov   = dependent_view->fov / 57.2957795f + 0.2f;
        splits[i].ratio = dependent_view->ratio;
    }

    glm::vec3 dir       = dir_light.transform.forward();
    glm::vec3 center    = dependent_view->position + dependent_view->direction * 50.0f;
    glm::vec3 light_pos = center - dir * ((dependent_view->far_plane - dependent_view->near_plane) / 2.0f);
    glm::vec3 right     = glm::cross(dir, glm::vec3(0.0f, 1.0f, 0.0f));

    glm::vec3 up = m_settings.pssm ? dependent_view->up : dependent_view->right;

    glm::mat4 modelview = glm::lookAt(light_pos, center, up);

    // Update splits

    {
        float nd = dependent_view->near_plane;
        float fd = dependent_view->far_plane;

        float lambda         = m_settings.csm_lambda;
        float ratio          = fd / nd;
        splits[0].near_plane = nd;

        for (int i = 1; i < count; i++)
        {
            float si = i / float(m_settings.cascade_count);

            // Practical Split Scheme: https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch10.html
            float t_near            = lambda * (nd * pow(ratio, si)) + (1 - lambda) * (nd + (fd - nd) * si);
            float t_far             = t_near * 1.005;
            splits[i].near_plane    = t_near;
            splits[i - 1].far_plane = t_far;
        }

        splits[m_settings.cascade_count - 1].far_plane = fd;
    }

    // Update frustum corners

    {
        glm::vec3 center   = dependent_view->position;
        glm::vec3 view_dir = dependent_view->direction;

        glm::vec3 up    = glm::vec3(0.0f, 1.0f, 0.0f);
        glm::vec3 right = glm::cross(view_dir, up);

        for (int i = 0; i < m_settings.cascade_count; i++)
        {
            glm::vec3 fc = center + view_dir * splits[i].far_plane;
            glm::vec3 nc = center + view_dir * splits[i].near_plane;

            right = glm::normalize(right);
            up    = glm::normalize(glm::cross(right, view_dir));

            // these heights and widths are half the heights and widths of
            // the near and far plane rectangles
            float near_height = tan(splits[i].fov / 2.0f) * splits[i].near_plane;
            float near_width  = near_height * splits[i].ratio;
            float far_height  = tan(splits[i].fov / 2.0f) * splits[i].far_plane;
            float far_width   = far_height * splits[i].ratio;

            splits[i].corners[0] = nc - up * near_height - right * near_width; // near-bottom-left
            splits[i].corners[1] = nc + up * near_height - right * near_width; // near-top-left
            splits[i].corners[2] = nc + up * near_height + right * near_width; // near-top-right
            splits[i].corners[3] = nc - up * near_height + right * near_width; // near-bottom-right

            splits[i].corners[4] = fc - up * far_height - right * far_width; // far-bottom-left
            splits[i].corners[5] = fc + up * far_height - right * far_width; // far-top-left
            splits[i].corners[6] = fc + up * far_height + right * far_width; // far-top-right
            splits[i].corners[7] = fc - up * far_height + right * far_width; // far-bottom-right
        }

        if (parent)
        {
            splits[m_settings.cascade_count].corners[0] = splits[0].corners[0]; // near-bottom-left
            splits[m_settings.cascade_count].corners[1] = splits[0].corners[1]; // near-top-left
            splits[m_settings.cascade_count].corners[2] = splits[0].corners[2]; // near-top-right
            splits[m_settings.cascade_count].corners[3] = splits[0].corners[3]; // near-bottom-right

            splits[m_settings.cascade_count].corners[4] = splits[m_settings.cascade_count - 1].corners[4]; // far-bottom-left
            splits[m_settings.cascade_count].corners[5] = splits[m_settings.cascade_count - 1].corners[5]; // far-top-left
            splits[m_settings.cascade_count].corners[6] = splits[m_settings.cascade_count - 1].corners[6]; // far-top-right
            splits[m_settings.cascade_count].corners[7] = splits[m_settings.cascade_count - 1].corners[7]; // far-bottom-right
        }
    }

    // Update crop matrices

    {
        glm::mat4 t_projection;

        for (int i = 0; i < count; i++)
        {
            glm::vec3 tmax = glm::vec3(-INFINITY, -INFINITY, -INFINITY);
            glm::vec3 tmin = glm::vec3(INFINITY, INFINITY, INFINITY);

            // find the z-range of the current frustum as seen from the light
            // in order to increase precision

            // note that only the z-component is need and thus
            // the multiplication can be simplified
            // transf.z = shad_modelview[2] * f.point[0].x + shad_modelview[6] * f.point[0].y + shad_modelview[10] * f.point[0].z + shad_modelview[14];
            glm::vec4 t_transf = modelview * glm::vec4(splits[i].corners[0], 1.0f);

            tmin.z = t_transf.z;
            tmax.z = t_transf.z;

            for (int j = 1; j < 8; j++)
            {
                t_transf = modelview * glm::vec4(splits[i].corners[j], 1.0f);
                if (t_transf.z > tmax.z)
                    tmax.z = t_transf.z;
                if (t_transf.z < tmin.z)
                    tmin.z = t_transf.z;
            }

            //tmax.z += 50; // TODO: This solves the dissapearing shadow problem. but how to fix?

            // Calculate frustum split center
            splits[i].center = glm::vec3(0.0f, 0.0f, 0.0f);

            for (int j = 0; j < 8; j++)
                splits[i].center += splits[i].corners[j];

            splits[i].center /= 8.0;

            if (m_settings.pssm)
            {
                // Calculate bounding sphere radius
                float radius = 0.0;

                for (int j = 0; j < 8; j++)
                {
                    float l = length(splits[i].corners[j] - splits[i].center);
                    radius  = std::max(radius, l);
                }

                radius = ceil(radius * 16.0) / 16.0;

                // Find bounding box that fits the sphere
                glm::vec3 radius3 = glm::vec3(radius, radius, radius);

                glm::vec3 max = radius3;
                glm::vec3 min = -radius3;

                glm::vec3 cascade_extents = max - min;

                // Push the light position back along the light direction by the near offset.
                glm::vec3 shadow_camera_pos = splits[i].center - dir * dependent_view->far_plane;

                // Add the near offset to the Z value of the cascade extents to make sure the orthographic frustum captures the entire frustum split (else it will exhibit cut-off issues).
                glm::mat4 ortho = glm::ortho(min.x, max.x, min.y, max.y, -dependent_view->far_plane, dependent_view->far_plane + cascade_extents.z);
                glm::mat4 view  = glm::lookAt(shadow_camera_pos, splits[i].center, dependent_view->up);

                proj_matrices[i] = ortho;
                crop[i]          = ortho * view;

                glm::vec4 shadow_origin = glm::vec4(0.0, 0.0, 0.0, 1.0);
                shadow_origin           = crop[i] * shadow_origin;
                shadow_origin           = shadow_origin * (kDirectionalLightShadowMapSizes[m_settings.shadow_map_quality] / 2.0f);

                glm::vec4 rounded_origin = round(shadow_origin);
                glm::vec4 round_offset   = rounded_origin - shadow_origin;
                round_offset             = round_offset * (2.0f / kDirectionalLightShadowMapSizes[m_settings.shadow_map_quality]);
                round_offset.z           = 0.0;
                round_offset.w           = 0.0;

                glm::mat4 shadow_proj = proj_matrices[i];

                shadow_proj[3][0] += round_offset.x;
                shadow_proj[3][1] += round_offset.y;
                shadow_proj[3][2] += round_offset.z;
                shadow_proj[3][3] += round_offset.w;

                if (!parent)
                {
                    cascade_views[i]->view_mat       = view;
                    cascade_views[i]->projection_mat = shadow_proj;
                    cascade_views[i]->vp_mat         = shadow_proj * view;
                }
                else
                {
                    parent->view_mat       = view;
                    parent->projection_mat = shadow_proj;
                    parent->vp_mat         = shadow_proj * view;
                }
            }
            else
            {
                glm::mat4 t_ortho    = glm::ortho(-1.0f, 1.0f, -1.0f, 1.0f, -dependent_view->far_plane, -tmin.z);
                glm::mat4 t_shad_mvp = t_ortho * modelview;

                // find the extends of the frustum slice as projected in light's homogeneous coordinates
                for (int j = 0; j < 8; j++)
                {
                    t_transf = t_shad_mvp * glm::vec4(splits[i].corners[j], 1.0);

                    t_transf.x /= t_transf.w;
                    t_transf.y /= t_transf.w;

                    if (t_transf.x > tmax.x) { tmax.x = t_transf.x; }
                    if (t_transf.x < tmin.x) { tmin.x = t_transf.x; }
                    if (t_transf.y > tmax.y) { tmax.y = t_transf.y; }
                    if (t_transf.y < tmin.y) { tmin.y = t_transf.y; }
                }

                glm::vec2 tscale  = glm::vec2(2.0 / (tmax.x - tmin.x), 2.0 / (tmax.y - tmin.y));
                glm::vec2 toffset = glm::vec2(-0.5 * (tmax.x + tmin.x) * tscale.x, -0.5 * (tmax.y + tmin.y) * tscale.y);

                glm::mat4 t_shad_crop = glm::mat4(1.0);
                t_shad_crop[0][0]     = tscale.x;
                t_shad_crop[1][1]     = tscale.y;
                t_shad_crop[0][3]     = toffset.x;
                t_shad_crop[1][3]     = toffset.y;
                t_shad_crop           = glm::transpose(t_shad_crop);

                t_projection = t_shad_crop * t_ortho;

                // Store the projection matrix
                proj_matrices[i] = t_projection;

                if (!parent)
                {
                    cascade_views[i]->view_mat       = modelview;
                    cascade_views[i]->projection_mat = t_projection;
                    cascade_views[i]->vp_mat         = t_projection * modelview;
                }
                else
                {
                    parent->view_mat       = modelview;
                    parent->projection_mat = t_projection;
                    parent->vp_mat         = t_projection * modelview;
                }
            }
        }

        // Update texture matrices

        for (int i = 0; i < m_settings.cascade_count; i++)
        {
            dependent_view->cascade_matrix[i] = bias * cascade_views[i]->vp_mat;

            // f[i].fard is originally in eye space - tell's us how far we can see.
            // Here we compute it in camera homogeneous coordinates. Basically, we calculate
            // cam_proj * (0, 0, f[i].fard, 1)^t and then normalize to [0; 1]

            glm::vec4 pos = dependent_view->projection_mat * glm::vec4(0.0, 0.0, -splits[i].far_plane, 1.0);
            glm::vec4 ndc = pos / pos.w;

            dependent_view->cascade_far_plane[i] = ndc.z * 0.5 + 0.5;
        }
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Renderer::create_cube()
{
    float cube_vertices[] = {
        // back face
        -1.0f,
        -1.0f,
        -1.0f,
        0.0f,
        0.0f,
        -1.0f,
        0.0f,
        0.0f, // bottom-left
        1.0f,
        1.0f,
        -1.0f,
        0.0f,
        0.0f,
        -1.0f,
        1.0f,
        1.0f, // top-right
        1.0f,
        -1.0f,
        -1.0f,
        0.0f,
        0.0f,
        -1.0f,
        1.0f,
        0.0f, // bottom-right
        1.0f,
        1.0f,
        -1.0f,
        0.0f,
        0.0f,
        -1.0f,
        1.0f,
        1.0f, // top-right
        -1.0f,
        -1.0f,
        -1.0f,
        0.0f,
        0.0f,
        -1.0f,
        0.0f,
        0.0f, // bottom-left
        -1.0f,
        1.0f,
        -1.0f,
        0.0f,
        0.0f,
        -1.0f,
        0.0f,
        1.0f, // top-left
        // front face
        -1.0f,
        -1.0f,
        1.0f,
        0.0f,
        0.0f,
        1.0f,
        0.0f,
        0.0f, // bottom-left
        1.0f,
        -1.0f,
        1.0f,
        0.0f,
        0.0f,
        1.0f,
        1.0f,
        0.0f, // bottom-right
        1.0f,
        1.0f,
        1.0f,
        0.0f,
        0.0f,
        1.0f,
        1.0f,
        1.0f, // top-right
        1.0f,
        1.0f,
        1.0f,
        0.0f,
        0.0f,
        1.0f,
        1.0f,
        1.0f, // top-right
        -1.0f,
        1.0f,
        1.0f,
        0.0f,
        0.0f,
        1.0f,
        0.0f,
        1.0f, // top-left
        -1.0f,
        -1.0f,
        1.0f,
        0.0f,
        0.0f,
        1.0f,
        0.0f,
        0.0f, // bottom-left
        // left face
        -1.0f,
        1.0f,
        1.0f,
        -1.0f,
        0.0f,
        0.0f,
        1.0f,
        0.0f, // top-right
        -1.0f,
        1.0f,
        -1.0f,
        -1.0f,
        0.0f,
        0.0f,
        1.0f,
        1.0f, // top-left
        -1.0f,
        -1.0f,
        -1.0f,
        -1.0f,
        0.0f,
        0.0f,
        0.0f,
        1.0f, // bottom-left
        -1.0f,
        -1.0f,
        -1.0f,
        -1.0f,
        0.0f,
        0.0f,
        0.0f,
        1.0f, // bottom-left
        -1.0f,
        -1.0f,
        1.0f,
        -1.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f, // bottom-right
        -1.0f,
        1.0f,
        1.0f,
        -1.0f,
        0.0f,
        0.0f,
        1.0f,
        0.0f, // top-right
        // right face
        1.0f,
        1.0f,
        1.0f,
        1.0f,
        0.0f,
        0.0f,
        1.0f,
        0.0f, // top-left
        1.0f,
        -1.0f,
        -1.0f,
        1.0f,
        0.0f,
        0.0f,
        0.0f,
        1.0f, // bottom-right
        1.0f,
        1.0f,
        -1.0f,
        1.0f,
        0.0f,
        0.0f,
        1.0f,
        1.0f, // top-right
        1.0f,
        -1.0f,
        -1.0f,
        1.0f,
        0.0f,
        0.0f,
        0.0f,
        1.0f, // bottom-right
        1.0f,
        1.0f,
        1.0f,
        1.0f,
        0.0f,
        0.0f,
        1.0f,
        0.0f, // top-left
        1.0f,
        -1.0f,
        1.0f,
        1.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f, // bottom-left
        // bottom face
        -1.0f,
        -1.0f,
        -1.0f,
        0.0f,
        -1.0f,
        0.0f,
        0.0f,
        1.0f, // top-right
        1.0f,
        -1.0f,
        -1.0f,
        0.0f,
        -1.0f,
        0.0f,
        1.0f,
        1.0f, // top-left
        1.0f,
        -1.0f,
        1.0f,
        0.0f,
        -1.0f,
        0.0f,
        1.0f,
        0.0f, // bottom-left
        1.0f,
        -1.0f,
        1.0f,
        0.0f,
        -1.0f,
        0.0f,
        1.0f,
        0.0f, // bottom-left
        -1.0f,
        -1.0f,
        1.0f,
        0.0f,
        -1.0f,
        0.0f,
        0.0f,
        0.0f, // bottom-right
        -1.0f,
        -1.0f,
        -1.0f,
        0.0f,
        -1.0f,
        0.0f,
        0.0f,
        1.0f, // top-right
        // top face
        -1.0f,
        1.0f,
        -1.0f,
        0.0f,
        1.0f,
        0.0f,
        0.0f,
        1.0f, // top-left
        1.0f,
        1.0f,
        1.0f,
        0.0f,
        1.0f,
        0.0f,
        1.0f,
        0.0f, // bottom-right
        1.0f,
        1.0f,
        -1.0f,
        0.0f,
        1.0f,
        0.0f,
        1.0f,
        1.0f, // top-right
        1.0f,
        1.0f,
        1.0f,
        0.0f,
        1.0f,
        0.0f,
        1.0f,
        0.0f, // bottom-right
        -1.0f,
        1.0f,
        -1.0f,
        0.0f,
        1.0f,
        0.0f,
        0.0f,
        1.0f, // top-left
        -1.0f,
        1.0f,
        1.0f,
        0.0f,
        1.0f,
        0.0f,
        0.0f,
        0.0f // bottom-left
    };

    m_cube_vbo = std::make_shared<VertexBuffer>(GL_STATIC_DRAW, sizeof(cube_vertices), (void*)cube_vertices);

    VertexAttrib attribs[] = {
        {
            3,
            GL_FLOAT,
            false,
            0,
        },
        { 3, GL_FLOAT, false, sizeof(float) * 3 },
        { 2, GL_FLOAT, false, sizeof(float) * 6 }
    };

    m_cube_vao = std::make_shared<VertexArray>(m_cube_vbo.get(), nullptr, sizeof(float) * 8, 3, attribs);

    if (!m_cube_vbo || !m_cube_vao)
    {
        NIMBLE_LOG_FATAL("Failed to create Vertex Buffers/Arrays");
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

int32_t Renderer::find_render_target_last_usage(std::shared_ptr<RenderTarget> rt)
{
    int32_t node_gid     = 0;
    int32_t last_node_id = -1;

    for (uint32_t graph_idx = 0; graph_idx < m_registered_render_graphs.size(); graph_idx++)
    {
        std::shared_ptr<RenderGraph> graph = m_registered_render_graphs[graph_idx];

        for (uint32_t node_idx = 0; node_idx < graph->node_count(); node_idx++)
        {
            std::shared_ptr<RenderNode> node = graph->node(node_idx);

            for (uint32_t rt_idx = 0; rt_idx < node->input_render_target_count(); rt_idx++)
            {
                std::shared_ptr<RenderTarget> input_rt = node->input_render_target(rt_idx);

                if (input_rt)
                {
                    if (rt->id == input_rt->id)
                        last_node_id = node_gid;
                }
            }

            node_gid++;
        }
    }

    return last_node_id;
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool Renderer::is_aliasing_candidate(std::shared_ptr<RenderTarget> rt, uint32_t write_node, uint32_t read_node, const RenderTargetDesc& rt_desc)
{
    bool format_match = rt->internal_format == rt_desc.rt->texture->internal_format() && rt->target == rt_desc.rt->texture->target() && rt->scale_h == rt_desc.rt->scale_h && rt->scale_w == rt_desc.rt->scale_w && rt->w == rt_desc.rt->w && rt->h == rt_desc.rt->h;

    if (!format_match)
        return false;

    for (auto& pair : rt_desc.lifetimes)
    {
        // Is this an intermediate texture?
        if (write_node == read_node)
        {
            if (write_node == pair.first || write_node == pair.second)
                return false;
        }
        else
        {
            if (write_node == pair.first || write_node == pair.second || read_node == pair.first || read_node == pair.second || (write_node > pair.first && read_node < pair.second))
                return false;
        }
    }

    return true;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Renderer::create_texture_for_render_target(std::shared_ptr<RenderTarget> rt, uint32_t write_node, uint32_t read_node)
{
    // Create new texture
    std::shared_ptr<Texture> tex;

    if (rt->is_scaled())
    {
        rt->w = uint32_t(rt->scale_w * float(m_window_width));
        rt->h = uint32_t(rt->scale_h * float(m_window_height));
    }

    if (rt->target == GL_TEXTURE_2D)
        tex = std::make_shared<Texture2D>(rt->w, rt->h, rt->array_size, rt->mip_levels, rt->num_samples, rt->internal_format, rt->format, rt->type);
    else if (rt->target == GL_TEXTURE_CUBE_MAP)
        tex = std::make_shared<TextureCube>(rt->w, rt->h, rt->array_size, rt->mip_levels, rt->internal_format, rt->format, rt->type);

    tex->set_wrapping(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);

    // Assign it to the current output Render Target
    rt->texture = tex;

    // Push it into the list of total Render Targets
    m_rt_cache.push_back({ rt, { { write_node, read_node } } });
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Renderer::bake_render_graphs()
{
    uint32_t node_gid = 0;

    for (uint32_t graph_idx = 0; graph_idx < m_registered_render_graphs.size(); graph_idx++)
    {
        std::shared_ptr<RenderGraph> graph = m_registered_render_graphs[graph_idx];

        for (uint32_t node_idx = 0; node_idx < graph->node_count(); node_idx++)
        {
            std::shared_ptr<RenderNode> node = graph->node(node_idx);

            for (uint32_t rt_idx = 0; rt_idx < node->output_render_target_count(); rt_idx++)
            {
                std::shared_ptr<RenderTarget> rt = node->output_render_target(rt_idx);

                if (!rt)
                    continue;

                if (rt->forward_slot == "")
                {
                    // Find last usage of output
                    int32_t current_node_id = node_gid;
                    int32_t last_node_id    = find_render_target_last_usage(rt);

                    bool found_texture = false;

                    // Try to find an already created texture that does not have an overlapping lifetime
                    for (auto& desc : m_rt_cache)
                    {
                        // Check if current Texture is suitable to be aliased
                        if (is_aliasing_candidate(rt, current_node_id, last_node_id, desc))
                        {
                            found_texture = true;
                            // Add the new lifetime to the existing texture
                            desc.lifetimes.push_back({ current_node_id, last_node_id });
                            rt->texture = desc.rt->texture;
                        }
                    }

                    if (!found_texture)
                        create_texture_for_render_target(rt, current_node_id, last_node_id);
                }
                else
                {
                    // Forwarded render target
                    auto input_rt = node->find_input_render_target(rt->forward_slot);

                    if (input_rt)
                    {
                        rt->id              = input_rt->id;
                        rt->scale_w         = input_rt->scale_w;
                        rt->scale_h         = input_rt->scale_h;
                        rt->w               = input_rt->w;
                        rt->h               = input_rt->h;
                        rt->target          = input_rt->target;
                        rt->internal_format = input_rt->internal_format;
                        rt->format          = input_rt->format;
                        rt->type            = input_rt->type;
                        rt->num_samples     = input_rt->num_samples;
                        rt->array_size      = input_rt->array_size;
                        rt->mip_levels      = input_rt->mip_levels;
                        rt->texture         = input_rt->texture;
                    }
                }
            }

            for (uint32_t rt_idx = 0; rt_idx < node->intermediate_render_target_count(); rt_idx++)
            {
                std::shared_ptr<RenderTarget> rt = node->intermediate_render_target(rt_idx);

                bool found_texture = false;

                // Try to find an already created texture that does not have an overlapping lifetime
                for (auto& desc : m_rt_cache)
                {
                    // Check if current Texture is suitable to be aliased
                    if (is_aliasing_candidate(rt, node_gid, node_gid, desc))
                    {
                        found_texture = true;
                        // Add the new lifetime to the existing texture
                        desc.lifetimes.push_back({ node_gid, node_gid });
                        rt->texture = desc.rt->texture;
                    }
                }

                if (!found_texture)
                    create_texture_for_render_target(rt, node_gid, node_gid);
            }

            node_gid++;
        }
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Renderer::update_uniforms(double delta)
{
    if (!m_scene.expired())
    {
        auto scene = m_scene.lock();

        // Update per entity uniforms
        Entity* entities = scene->entities();

        for (uint32_t i = 0; i < scene->entity_count(); i++)
        {
            Entity& entity = entities[i];

            m_per_entity_uniforms[i].modal_mat      = entity.transform.model;
            m_per_entity_uniforms[i].last_model_mat = entity.transform.prev_model;
        }

        void* ptr = m_per_entity->map(GL_WRITE_ONLY);
        memcpy(ptr, &m_per_entity_uniforms[0], sizeof(PerEntityUniforms) * scene->entity_count());
        m_per_entity->unmap();

        // Update per view uniforms
        for (uint32_t i = 0; i < m_num_update_views; i++)
        {
            View* view = m_update_views[i];

            m_per_view_uniforms[i].view_mat            = view->view_mat;
            m_per_view_uniforms[i].proj_mat            = view->projection_mat;
            m_per_view_uniforms[i].view_proj           = view->vp_mat;
            m_per_view_uniforms[i].last_view_proj      = view->prev_vp_mat;
            m_per_view_uniforms[i].inv_proj            = view->inv_projection_mat;
            m_per_view_uniforms[i].inv_view            = view->inv_view_mat;
            m_per_view_uniforms[i].inv_view_proj       = view->inv_vp_mat;
            m_per_view_uniforms[i].view_pos            = glm::vec4(view->position, 0.0f);
            m_per_view_uniforms[i].view_dir            = glm::vec4(view->direction, 0.0f);
            m_per_view_uniforms[i].near_plane          = view->near_plane;
            m_per_view_uniforms[i].far_plane           = view->far_plane;
            m_per_view_uniforms[i].num_cascades        = m_settings.cascade_count;
            m_per_view_uniforms[i].viewport_width      = m_window_width;
            m_per_view_uniforms[i].viewport_height     = m_window_height;
            m_per_view_uniforms[i].current_prev_jitter = view->jitter;
            m_per_view_uniforms[i].time_params         = glm::vec4(static_cast<float>(glfwGetTime()), sinf(static_cast<float>(glfwGetTime())), cosf(static_cast<float>(glfwGetTime())), static_cast<float>(delta));
            m_per_view_uniforms[i].viewport_params     = glm::vec4(static_cast<float>(view->viewport_width), static_cast<float>(view->viewport_height), 1.0f / static_cast<float>(view->viewport_width), 1.0f / static_cast<float>(view->viewport_height));

            float z_buffer_params_x = -1.0 + (view->near_plane / view->far_plane);

            m_per_view_uniforms[i].z_buffer_params = glm::vec4(z_buffer_params_x, 1.0f, z_buffer_params_x / view->near_plane, 1.0f / view->near_plane);

            for (uint32_t j = 0; j < view->num_cascade_frustums; j++)
            {
                m_per_view_uniforms[i].cascade_matrix[j]    = view->cascade_matrix[j];
                m_per_view_uniforms[i].cascade_far_plane[j] = view->cascade_far_plane[j];
            }
        }

        ptr = m_per_view->map(GL_WRITE_ONLY);
        memcpy(ptr, &m_per_view_uniforms[0], sizeof(PerViewUniforms) * m_num_update_views);
        m_per_view->unmap();

        // Update per scene uniforms
        DirectionalLight* dir_lights = scene->directional_lights();

        m_per_scene_uniforms.directional_light_count = scene->directional_light_count();

        for (int32_t light_idx = 0; light_idx < m_per_scene_uniforms.directional_light_count; light_idx++)
        {
            DirectionalLight& light = dir_lights[light_idx];

            m_per_scene_uniforms.shadow_map_bias[light_idx].x                 = light.shadow_map_bias;
            m_per_scene_uniforms.directional_light_direction[light_idx]       = glm::vec4(light.transform.forward(), 0.0f);
            m_per_scene_uniforms.directional_light_color_intensity[light_idx] = glm::vec4(light.color, light.intensity);
            m_per_scene_uniforms.directional_light_casts_shadow[light_idx]    = light.casts_shadow ? 1 : 0;
        }

        SpotLight* spot_lights = scene->spot_lights();

        m_per_scene_uniforms.spot_light_count = scene->spot_light_count();

        for (int32_t light_idx = 0; light_idx < m_per_scene_uniforms.spot_light_count; light_idx++)
        {
            SpotLight& light = spot_lights[light_idx];

            m_per_scene_uniforms.shadow_map_bias[light_idx].y             = light.shadow_map_bias;
            m_per_scene_uniforms.spot_light_direction_range[light_idx]    = glm::vec4(light.transform.forward(), light.range);
            m_per_scene_uniforms.spot_light_color_intensity[light_idx]    = glm::vec4(light.color, light.intensity);
            m_per_scene_uniforms.spot_light_position[light_idx]           = glm::vec4(light.transform.position, 0.0f);
            m_per_scene_uniforms.spot_light_casts_shadow[light_idx]       = light.casts_shadow ? 1 : 0;
            m_per_scene_uniforms.spot_light_cutoff_inner_outer[light_idx] = glm::vec4(cosf(glm::radians(light.inner_cone_angle)), cosf(glm::radians(light.outer_cone_angle)), 0.0f, 0.0f);
        }

        PointLight* point_lights = scene->point_lights();

        m_per_scene_uniforms.point_light_count = scene->point_light_count();

        for (int32_t light_idx = 0; light_idx < m_per_scene_uniforms.point_light_count; light_idx++)
        {
            PointLight& light = point_lights[light_idx];

            m_per_scene_uniforms.shadow_map_bias[light_idx].z           = light.shadow_map_bias;
            m_per_scene_uniforms.point_light_position[light_idx]        = glm::vec4(light.transform.position, 0.0f);
            m_per_scene_uniforms.point_light_near_far[light_idx]        = glm::vec4(POINT_LIGHT_NEAR_PLANE, light.range, 0.0f, 0.0f);
            m_per_scene_uniforms.point_light_color_intensity[light_idx] = glm::vec4(light.color, light.intensity);
            m_per_scene_uniforms.point_light_casts_shadow[light_idx]    = light.casts_shadow ? 1 : 0;
        }

        ptr = m_per_scene->map(GL_WRITE_ONLY);
        memcpy(ptr, &m_per_scene_uniforms, sizeof(PerSceneUniforms));
        m_per_scene->unmap();
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Renderer::cull_scene()
{
    NIMBLE_SCOPED_SAMPLE(PROFILER_FRUSTUM_CULLING);

    if (!m_scene.expired())
    {
        auto scene = m_scene.lock();

        Entity* entities = scene->entities();

        for (uint32_t i = 0; i < scene->entity_count(); i++)
        {
            Entity& entity = entities[i];

            entity.obb.position    = entity.transform.position;
            entity.obb.orientation = glm::mat3(entity.transform.model);

            for (uint32_t j = 0; j < m_num_cull_views; j++)
            {
                if (intersects(m_active_frustums[j], entity.obb))
                {
                    entity.set_visible(j);

#ifdef ENABLE_SUBMESH_CULLING
                    for (uint32_t k = 0; k < entity.mesh->submesh_count(); k++)
                    {
                        SubMesh&  submesh            = entity.mesh->submesh(k);
                        glm::vec3 center             = (submesh.min_extents + submesh.max_extents) / 2.0f;
                        glm::vec4 transformed_center = entity.transform.model * glm::vec4(center, 1.0f);

                        entity.submesh_spheres[k].position = transformed_center;

                        if (intersects(m_active_frustums[j], entity.submesh_spheres[k]))
                            entity.set_submesh_visible(k, j);
                        else
                            entity.set_submesh_invisible(k, j);
                    }
#endif
                }
                else
                    entity.set_invisible(j);
            }
        }
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool Renderer::queue_rendered_view(View* view)
{
    if (m_num_rendered_views == MAX_VIEWS)
    {
        NIMBLE_LOG_ERROR("Maximum number of Views reached (64)");
        return false;
    }
    else
    {
        uint32_t rendered_idx          = m_num_rendered_views++;
        m_rendered_views[rendered_idx] = view;

        return true;
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

uint32_t Renderer::queue_update_view(View* view)
{
    if (m_num_update_views == MAX_VIEWS)
    {
        NIMBLE_LOG_ERROR("Maximum number of Views reached (64)");
        return UINT32_MAX;
    }
    else
    {
        uint32_t update_idx        = m_num_update_views++;
        m_update_views[update_idx] = view;

        return update_idx;
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

uint32_t Renderer::queue_culled_view(Frustum f)
{
    if (m_num_cull_views == MAX_VIEWS)
    {
        NIMBLE_LOG_ERROR("Maximum number of Views reached (64)");
        return UINT32_MAX;
    }
    else
    {
        uint32_t culled_idx           = m_num_cull_views++;
        m_active_frustums[culled_idx] = f;

        return culled_idx;
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Renderer::queue_default_views()
{
    if (!m_scene.expired())
    {
        auto scene = m_scene.lock();

        // Allocate view for scene camera
        auto  camera     = scene->camera();
        View* scene_view = allocate_view();

        scene_view->tag                                 = "Scene View";
        scene_view->enabled                             = true;
        scene_view->culling                             = true;
        scene_view->direction                           = camera->m_forward;
        scene_view->position                            = camera->m_position;
        scene_view->up                                  = camera->m_up;
        scene_view->right                               = camera->m_right;
        scene_view->view_mat                            = camera->m_view;
        scene_view->projection_mat                      = camera->m_projection;
        scene_view->vp_mat                              = camera->m_view_projection;
        scene_view->prev_vp_mat                         = camera->m_prev_view_projection;
        scene_view->inv_view_mat                        = glm::inverse(camera->m_view);
        scene_view->inv_projection_mat                  = glm::inverse(camera->m_projection);
        scene_view->inv_vp_mat                          = glm::inverse(camera->m_view_projection);
        scene_view->jitter                              = glm::vec4(camera->m_prev_jitter, camera->m_current_jitter);
        scene_view->dest_color_render_target_view_count = 0;
        scene_view->dest_color_render_target_views      = nullptr;
        scene_view->dest_depth_render_target_view       = nullptr;
        scene_view->viewport                            = camera->m_viewport;
        scene_view->graph                               = m_scene_render_graph;
        scene_view->type                                = VIEW_STANDARD;
        scene_view->fov                                 = camera->m_fov;
        scene_view->ratio                               = camera->m_aspect_ratio;
        scene_view->near_plane                          = camera->m_near;
        scene_view->far_plane                           = camera->m_far;
        scene_view->viewport_width                      = m_window_width;
        scene_view->viewport_height                     = m_window_height;

        // Queue shadow views
        queue_spot_light_views();
        queue_point_light_views();
        queue_directional_light_views(scene_view);

        // Finally queue the scene view
        queue_view(scene_view);
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Renderer::render_all_views(double delta)
{
    NIMBLE_SCOPED_SAMPLE("Render All Views");

    if (m_num_rendered_views > 0)
    {
        auto scene = m_scene.lock();

        for (uint32_t i = 0; i < m_num_rendered_views; i++)
        {
            View* view = m_rendered_views[i];

            if (view->enabled)
            {
                if (view->graph)
                {
                    NIMBLE_SCOPED_SAMPLE(view->tag);
                    view->graph->execute(delta, this, scene.get(), view);
                }
                else
                    NIMBLE_LOG_ERROR("Render Graph not assigned for View!");
            }
        }
    }
    else
        glClear(GL_COLOR_BUFFER_BIT);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Renderer::render_debug_output()
{
    if (!m_debug_render_target || !m_debug_render_target->texture)
        return;

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    m_debug_program->use();

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (m_scaled_debug_output)
        glViewport(0, 0, m_window_width, m_window_height);
    else
    {
        Texture2D* tex = (Texture2D*)m_debug_render_target->texture.get();
        glViewport(0, 0, tex->width(), tex->height());
    }

    if (m_debug_program->set_uniform("s_Texture", 0))
        m_debug_render_target->texture->bind(0);

    m_debug_program->set_uniform("u_Mask", m_debug_color_mask);

    // Render fullscreen triangle
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble