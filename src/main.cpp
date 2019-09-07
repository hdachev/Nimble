#include <iostream>
#include <fstream>
#include <gtc/matrix_transform.hpp>
#include <gtc/type_ptr.hpp>
#include <memory>
#include "application.h"
#include "camera.h"
#include "utility.h"
#include "material.h"
#include "macros.h"
#include "render_graph.h"
#include "nodes/forward_node.h"
#include "nodes/cubemap_skybox_node.h"
#include "nodes/pcf_point_light_depth_node.h"
#include "nodes/pcf_directional_light_depth_node.h"
#include "nodes/copy_node.h"
#include "nodes/g_buffer_node.h"
#include "nodes/deferred_node.h"
#include "nodes/tone_map_node.h"
#include "nodes/bloom_node.h"
#include "nodes/ssao_node.h"
#include "nodes/hiz_node.h"
#include "nodes/adaptive_exposure_node.h"
#include "nodes/motion_blur_node.h"
#include "nodes/volumetric_light_node.h"
#include "nodes/screen_space_reflection_node.h"
#include "nodes/reflection_node.h"
#include "nodes/fxaa_node.h"
#include "nodes/depth_of_field_node.h"
#include "debug_draw.h"
#include "imgui_helpers.h"
#include "external/nfd/nfd.h"
#include "profiler.h"
#include "probe_renderer/bruneton_probe_renderer.h"
#include "ImGuizmo.h"
#include <random>

#define NIMBLE_EDITOR

namespace nimble
{
#define CAMERA_DEFAULT_FOV 60.0f
#define CAMERA_DEFAULT_NEAR_PLANE 1.0f
#define CAMERA_DEFAULT_FAR_PLANE 6000.0f

class Nimble : public Application
{
protected:
    // -----------------------------------------------------------------------------------------------------------------------------------

    bool init(int argc, const char* argv[]) override
    {
        // Attempt to load startup scene.
        std::shared_ptr<Scene> scene = m_resource_manager.load_scene("scene/startup.json");

        if (scene)
            m_scene = scene;
        else
        {
            // If failed, prompt user to select scene to be loaded.
            if (!scene && !load_scene_from_dialog())
                return false;
        }

        //create_random_point_lights();
        //create_random_spot_lights();

        // Create camera.
        create_camera();

        create_render_graphs();

        return true;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void update(double delta) override
    {
        // Update camera.
        update_camera();

        if (m_debug_gui)
            gui();

#ifdef NIMBLE_EDITOR
        if (!m_edit_mode)
        {
#endif
            if (m_scene)
                m_scene->update();
#ifdef NIMBLE_EDITOR
        }
#endif
        m_renderer.render(delta, &m_viewport_manager);

        if (m_scene)
            m_debug_draw.render(nullptr, m_width, m_height, m_scene->camera()->m_view_projection);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void shutdown() override
    {
        m_forward_graph.reset();
        m_scene.reset();
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    AppSettings intial_app_settings() override
    {
        AppSettings settings;

        settings.resizable = true;
        settings.width     = 1280;
        settings.height    = 720;
        settings.title     = "Nimble - Dihara Wijetunga (c) 2019";

        return settings;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void window_resized(int width, int height) override
    {
        if (m_scene)
        {
            m_scene->camera()->m_width  = m_width;
            m_scene->camera()->m_height = m_height;

            // Override window resized method to update camera projection.
            m_scene->camera()->update_projection(m_scene->camera()->m_fov, m_scene->camera()->m_near, m_scene->camera()->m_far, float(m_width) / float(m_height));
        }

        m_viewport_manager.on_window_resized(width, height);
        m_renderer.on_window_resized(width, height);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void key_pressed(int code) override
    {
        // Handle forward movement.
        if (code == GLFW_KEY_W)
            m_heading_speed = m_camera_speed;
        else if (code == GLFW_KEY_S)
            m_heading_speed = -m_camera_speed;

        // Handle sideways movement.
        if (code == GLFW_KEY_A)
            m_sideways_speed = -m_camera_speed;
        else if (code == GLFW_KEY_D)
            m_sideways_speed = m_camera_speed;

        if (code == GLFW_KEY_G)
            m_debug_gui = !m_debug_gui;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void key_released(int code) override
    {
        // Handle forward movement.
        if (code == GLFW_KEY_W || code == GLFW_KEY_S)
            m_heading_speed = 0.0f;

        // Handle sideways movement.
        if (code == GLFW_KEY_A || code == GLFW_KEY_D)
            m_sideways_speed = 0.0f;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void mouse_pressed(int code) override
    {
        // Enable mouse look.
        if (code == GLFW_MOUSE_BUTTON_RIGHT)
            m_mouse_look = true;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void mouse_released(int code) override
    {
        // Disable mouse look.
        if (code == GLFW_MOUSE_BUTTON_RIGHT)
            m_mouse_look = false;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

private:
    // -----------------------------------------------------------------------------------------------------------------------------------

    void create_camera()
    {
        m_scene->camera()->m_width             = m_width;
        m_scene->camera()->m_height            = m_height;
        m_scene->camera()->m_half_pixel_jitter = false;
        m_scene->camera()->update_projection(CAMERA_DEFAULT_FOV, CAMERA_DEFAULT_NEAR_PLANE, CAMERA_DEFAULT_FAR_PLANE, float(m_width) / float(m_height));

        m_viewport = m_viewport_manager.create_viewport("Main", 0.0f, 0.0f, 1.0f, 1.0f, 0);

        m_scene->camera()->m_viewport = m_viewport;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void create_render_graphs()
    {
        REGISTER_RENDER_NODE(ForwardNode, m_resource_manager);
        REGISTER_RENDER_NODE(CubemapSkyboxNode, m_resource_manager);
        REGISTER_RENDER_NODE(PCFPointLightDepthNode, m_resource_manager);
        REGISTER_RENDER_NODE(PCFDirectionalLightDepthNode, m_resource_manager);
        REGISTER_RENDER_NODE(PCFSpotLightDepthNode, m_resource_manager);
        REGISTER_RENDER_NODE(CopyNode, m_resource_manager);
        REGISTER_RENDER_NODE(GBufferNode, m_resource_manager);
        REGISTER_RENDER_NODE(DeferredNode, m_resource_manager);
        REGISTER_RENDER_NODE(ToneMapNode, m_resource_manager);
        REGISTER_RENDER_NODE(BloomNode, m_resource_manager);
        REGISTER_RENDER_NODE(SSAONode, m_resource_manager);
        REGISTER_RENDER_NODE(HiZNode, m_resource_manager);
        REGISTER_RENDER_NODE(AdaptiveExposureNode, m_resource_manager);
        REGISTER_RENDER_NODE(MotionBlurNode, m_resource_manager);
        REGISTER_RENDER_NODE(VolumetricLightNode, m_resource_manager);
        REGISTER_RENDER_NODE(ScreenSpaceReflectionNode, m_resource_manager);
        REGISTER_RENDER_NODE(ReflectionNode, m_resource_manager);
        REGISTER_RENDER_NODE(FXAANode, m_resource_manager);
        REGISTER_RENDER_NODE(DepthOfFieldNode, m_resource_manager);

        // Create Forward render graph
        m_forward_graph = m_resource_manager.load_render_graph("graph/deferred_graph.json", &m_renderer);

        // Create Point Light render graph
        m_pcf_point_light_graph = m_resource_manager.load_shadow_render_graph("PCFPointLightDepthNode", &m_renderer);

        // Create Spot Light render graph
        m_pcf_spot_light_graph = m_resource_manager.load_shadow_render_graph("PCFSpotLightDepthNode", &m_renderer);

        // Create Directional Light render graph
        m_pcf_directional_light_graph = m_resource_manager.load_shadow_render_graph("PCFDirectionalLightDepthNode", &m_renderer);

        m_bruneton_probe_renderer = std::make_shared<BrunetonProbeRenderer>();

        // Set the graphs as the active graphs
        m_renderer.set_scene(m_scene);

        m_renderer.set_point_light_render_graph(m_pcf_point_light_graph);
        m_renderer.set_spot_light_render_graph(m_pcf_spot_light_graph);
        m_renderer.set_directional_light_render_graph(m_pcf_directional_light_graph);
        m_renderer.set_global_probe_renderer(m_bruneton_probe_renderer);

        m_renderer.set_scene_render_graph(m_forward_graph);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void create_random_spot_lights()
    {
        AABB aabb = m_scene->aabb();

        const float    range      = 300.0f;
        const float    intensity  = 10.0f;
        const uint32_t num_lights = 10;
        const float    aabb_scale = 0.6f;

        std::random_device rd;
        std::mt19937       gen(rd());

        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        std::uniform_real_distribution<float> dis_x(aabb.min.x * aabb_scale, aabb.max.x * aabb_scale);
        std::uniform_real_distribution<float> dis_y(aabb.min.y * aabb_scale, aabb.max.y * aabb_scale);
        std::uniform_real_distribution<float> dis_z(aabb.min.z * aabb_scale, aabb.max.z * aabb_scale);
        std::uniform_real_distribution<float> dis_pitch(0.0f, 180.0f);
        std::uniform_real_distribution<float> dis_yaw(0.0f, 360.0f);

        for (int n = 0; n < num_lights; n++)
            m_scene->create_spot_light(glm::vec3(dis_x(rd), dis_y(rd), dis_z(rd)), glm::vec3(dis_pitch(rd), dis_yaw(rd), 0.0f), glm::vec3(dis(rd), dis(rd), dis(rd)), 35.0f, 45.0f, 1000.0f, 10.0f);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void create_random_point_lights()
    {
        AABB aabb = m_scene->aabb();

        const float    range      = 300.0f;
        const float    intensity  = 10.0f;
        const uint32_t num_lights = 100;
        const float    aabb_scale = 0.6f;

        std::random_device rd;
        std::mt19937       gen(rd());

        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        std::uniform_real_distribution<float> dis_x(aabb.min.x * aabb_scale, aabb.max.x * aabb_scale);
        std::uniform_real_distribution<float> dis_y(aabb.min.y * aabb_scale, aabb.max.y * aabb_scale);
        std::uniform_real_distribution<float> dis_z(aabb.min.z * aabb_scale, aabb.max.z * aabb_scale);

        for (int n = 0; n < num_lights; n++)
            m_scene->create_point_light(glm::vec3(dis_x(rd), dis_y(rd), dis_z(rd)), glm::vec3(dis(rd), dis(rd), dis(rd)), range, intensity, false);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void gui()
    {
        float cpu_time = 0.0f;
        float gpu_time = 0.0f;
        ImGui::ShowDemoWindow();
        ImGuizmo::BeginFrame();

        ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
        ImGui::SetNextWindowSize(ImVec2(m_width, m_height));

        int flags = ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoCollapse;

        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImVec2(m_width, m_height));

        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);

        if (ImGui::Begin("Gizmo", (bool*)0, flags))
            ImGuizmo::SetDrawlist();
        ImGui::End();
        ImGui::PopStyleVar();

        if (m_edit_mode)
        {
            if (ImGui::Begin("Inspector"))
            {
                Transform* t = nullptr;

                if (m_selected_entity != UINT32_MAX)
                {
                    t = &m_scene->lookup_entity(m_selected_entity).transform;
                    edit_transform((float*)&m_scene->camera()->m_view, (float*)&m_scene->camera()->m_projection, t);
                }
                else if (m_selected_dir_light != UINT32_MAX)
                {
                    DirectionalLight& light = m_scene->lookup_directional_light(m_selected_dir_light);

                    t = &light.transform;
                    edit_transform((float*)&m_scene->camera()->m_view, (float*)&m_scene->camera()->m_projection, t, false, true, false);

                    ImGui::Separator();

                    ImGui::Checkbox("Casts Shadows", &light.casts_shadow);
                    ImGui::InputFloat("Shadow Map Bias", &light.shadow_map_bias);
                    ImGui::InputFloat("Intensity", &light.intensity);
                    ImGui::ColorPicker3("Color", &light.color.x);
                }
                else if (m_selected_point_light != UINT32_MAX)
                {
                    PointLight& light = m_scene->lookup_point_light(m_selected_point_light);

                    t = &light.transform;
                    edit_transform((float*)&m_scene->camera()->m_view, (float*)&m_scene->camera()->m_projection, t, true, false, false);

                    ImGui::Separator();

                    ImGui::Checkbox("Casts Shadows", &light.casts_shadow);
                    ImGui::InputFloat("Shadow Map Bias", &light.shadow_map_bias);
                    ImGui::InputFloat("Intensity", &light.intensity);
                    ImGui::InputFloat("Range", &light.range);
                    ImGui::ColorPicker3("Color", &light.color.x);

                    m_debug_draw.sphere(light.range, light.transform.position, light.color);
                }
                else if (m_selected_spot_light != UINT32_MAX)
                {
                    SpotLight& light = m_scene->lookup_spot_light(m_selected_spot_light);

                    t = &light.transform;
                    edit_transform((float*)&m_scene->camera()->m_view, (float*)&m_scene->camera()->m_projection, t, true, true, false);

                    ImGui::Separator();

                    ImGui::Checkbox("Casts Shadows", &light.casts_shadow);
                    ImGui::InputFloat("Shadow Map Bias", &light.shadow_map_bias);
                    ImGui::InputFloat("Intensity", &light.intensity);
                    ImGui::InputFloat("Range", &light.range);
                    ImGui::InputFloat("Inner Cone Angle", &light.inner_cone_angle);
                    ImGui::InputFloat("Outer Cone Angle", &light.outer_cone_angle);
                    ImGui::ColorPicker3("Color", &light.color.x);
                }
            }
            ImGui::End();
        }

        if (ImGui::Begin("Editor"))
        {
            ImGui::Checkbox("Edit Mode", &m_edit_mode);

            ImGui::Separator();

            if (ImGui::CollapsingHeader("Scene"))
            {
                if (m_scene)
                    ImGui::Text("Current Scene: %s", m_scene->name().c_str());
                else
                    ImGui::Text("Current Scene: -");

                if (ImGui::Button("Load"))
                {
                    if (load_scene_from_dialog())
                    {
                        create_camera();
                        m_renderer.set_scene(m_scene);
                    }
                }

                if (m_scene)
                {
                    if (ImGui::Button("Unload"))
                    {
                        m_scene = nullptr;
                        m_resource_manager.shutdown();
                    }
                }
            }

            if (ImGui::CollapsingHeader("Profiler"))
                profiler::ui();

            if (ImGui::CollapsingHeader("Render Graph"))
                render_node_params();

            if (ImGui::CollapsingHeader("Render Target Inspector"))
                render_target_inspector();

            if (m_renderer.global_probe_renderer())
            {
                if (ImGui::CollapsingHeader("Global Probe Renderer"))
                    paramerizable_ui(m_renderer.global_probe_renderer());
            }

            if (ImGui::CollapsingHeader("Entities"))
            {
                if (m_scene)
                {
                    Entity* entities = m_scene->entities();

                    for (uint32_t i = 0; i < m_scene->entity_count(); i++)
                    {
                        if (ImGui::Selectable(entities[i].name.c_str(), m_selected_entity == entities[i].id))
                        {
                            m_selected_entity      = entities[i].id;
                            m_selected_dir_light   = UINT32_MAX;
                            m_selected_point_light = UINT32_MAX;
                            m_selected_spot_light  = UINT32_MAX;
                        }
                    }
                }
            }

            if (ImGui::CollapsingHeader("Camera"))
            {
                std::shared_ptr<Camera> camera = m_scene->camera();

                float near_plane = camera->m_near;
                float far_plane  = camera->m_far;
                float fov        = camera->m_fov;

                ImGui::InputFloat("Near Plane", &near_plane);
                ImGui::InputFloat("Far Plane", &far_plane);
                ImGui::SliderFloat("FOV", &fov, 1.0f, 90.0f);

                if (near_plane != camera->m_near || far_plane != camera->m_far || fov != camera->m_fov)
                    camera->update_projection(fov, near_plane, far_plane, float(m_width) / float(m_height));

                ImGui::SliderFloat("Near Field Begin", &camera->m_near_begin, camera->m_near, camera->m_far);
                ImGui::SliderFloat("Near Field End", &camera->m_near_end, camera->m_near, camera->m_far);
                ImGui::SliderFloat("Far Field Begin", &camera->m_far_begin, camera->m_near, camera->m_far);
                ImGui::SliderFloat("Far Field End", &camera->m_far_end, camera->m_near, camera->m_far);
            }

            if (ImGui::CollapsingHeader("Point Lights"))
            {
                if (m_scene)
                {
                    PointLight* lights = m_scene->point_lights();

                    for (uint32_t i = 0; i < m_scene->point_light_count(); i++)
                    {
                        std::string name = std::to_string(lights[i].id);

                        if (ImGui::Selectable(name.c_str(), m_selected_point_light == lights[i].id))
                        {
                            m_selected_entity      = UINT32_MAX;
                            m_selected_dir_light   = UINT32_MAX;
                            m_selected_point_light = lights[i].id;
                            m_selected_spot_light  = UINT32_MAX;
                        }
                    }

                    ImGui::Separator();

                    ImGui::PushID(1);
                    if (ImGui::Button("Create"))
                        m_scene->create_point_light(glm::vec3(0.0f), glm::vec3(1.0f), 100.0f, 1.0f);

                    if (m_selected_point_light != UINT32_MAX)
                    {
                        ImGui::SameLine();

                        if (ImGui::Button("Remove"))
                        {
                            m_scene->destroy_point_light(m_selected_point_light);
                            m_selected_point_light = UINT32_MAX;
                        }
                    }

                    ImGui::PopID();
                }
            }

            if (ImGui::CollapsingHeader("Spot Lights"))
            {
                if (m_scene)
                {
                    SpotLight* lights = m_scene->spot_lights();

                    for (uint32_t i = 0; i < m_scene->spot_light_count(); i++)
                    {
                        std::string name = std::to_string(lights[i].id);

                        if (ImGui::Selectable(name.c_str(), m_selected_spot_light == lights[i].id))
                        {
                            m_selected_entity      = UINT32_MAX;
                            m_selected_dir_light   = UINT32_MAX;
                            m_selected_point_light = UINT32_MAX;
                            m_selected_spot_light  = lights[i].id;
                        }
                    }

                    ImGui::Separator();

                    ImGui::PushID(2);
                    if (ImGui::Button("Create"))
                        m_scene->create_spot_light(glm::vec3(0.0f), glm::vec3(0.0f), glm::vec3(1.0f), 35.0f, 45.0f, 100.0f, 1.0f);

                    if (m_selected_spot_light != UINT32_MAX)
                    {
                        ImGui::SameLine();

                        if (ImGui::Button("Remove"))
                        {
                            m_scene->destroy_spot_light(m_selected_spot_light);
                            m_selected_spot_light = UINT32_MAX;
                        }
                    }

                    ImGui::PopID();
                }
            }

            if (ImGui::CollapsingHeader("Directional Lights"))
            {
                if (m_scene)
                {
                    DirectionalLight* lights = m_scene->directional_lights();

                    for (uint32_t i = 0; i < m_scene->directional_light_count(); i++)
                    {
                        std::string name = std::to_string(lights[i].id);

                        if (ImGui::Selectable(name.c_str(), m_selected_dir_light == lights[i].id))
                        {
                            m_selected_entity      = UINT32_MAX;
                            m_selected_dir_light   = lights[i].id;
                            m_selected_point_light = UINT32_MAX;
                            m_selected_spot_light  = UINT32_MAX;
                        }
                    }

                    ImGui::Separator();

                    ImGui::PushID(3);
                    if (ImGui::Button("Create"))
                        m_scene->create_directional_light(glm::vec3(0.0f), glm::vec3(1.0f), 10.0f);

                    if (m_selected_dir_light != UINT32_MAX)
                    {
                        ImGui::SameLine();

                        if (ImGui::Button("Remove"))
                        {
                            m_scene->destroy_directional_light(m_selected_dir_light);
                            m_selected_dir_light = UINT32_MAX;
                        }
                    }

                    ImGui::PopID();
                }
            }
        }

        ImGui::End();
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void paramerizable_ui(std::shared_ptr<Parameterizable> paramterizable)
    {
        int32_t num_bool_params  = 0;
        int32_t num_int_params   = 0;
        int32_t num_float_params = 0;

        BoolParameter*  bool_params  = paramterizable->bool_parameters(num_bool_params);
        IntParameter*   int_params   = paramterizable->int_parameters(num_int_params);
        FloatParameter* float_params = paramterizable->float_parameters(num_float_params);

        for (uint32_t i = 0; i < num_bool_params; i++)
            ImGui::Checkbox(bool_params[i].name.c_str(), bool_params[i].ptr);

        for (uint32_t i = 0; i < num_int_params; i++)
        {
            if (int_params[i].min == int_params[i].max)
                ImGui::InputInt(int_params[i].name.c_str(), int_params[i].ptr);
            else
                ImGui::SliderInt(int_params[i].name.c_str(), int_params[i].ptr, int_params[i].min, int_params[i].max);
        }

        for (uint32_t i = 0; i < num_float_params; i++)
        {
            if (float_params[i].min == float_params[i].max)
                ImGui::InputFloat(float_params[i].name.c_str(), float_params[i].ptr);
            else
                ImGui::SliderFloat(float_params[i].name.c_str(), float_params[i].ptr, float_params[i].min, float_params[i].max);
        }
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void render_node_params()
    {
        for (uint32_t i = 0; i < m_forward_graph->node_count(); i++)
        {
            auto node = m_forward_graph->node(i);

            if (ImGui::TreeNode(node->name().c_str()))
            {
                paramerizable_ui(node);
                ImGui::TreePop();
            }
        }
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void render_target_inspector()
    {
        bool scaled = m_renderer.scaled_debug_output();
        ImGui::Checkbox("Scaled Debug Output", &scaled);
        m_renderer.set_scaled_debug_output(scaled);

        bool      mask_bool[4];
        glm::vec4 mask = m_renderer.debug_color_mask();

        mask_bool[0] = (bool)mask.x;
        mask_bool[1] = (bool)mask.y;
        mask_bool[2] = (bool)mask.z;
        mask_bool[3] = (bool)mask.w;

        ImGui::Checkbox("Red", &mask_bool[0]);
        ImGui::Checkbox("Green", &mask_bool[1]);
        ImGui::Checkbox("Blue", &mask_bool[2]);
        ImGui::Checkbox("Alpha", &mask_bool[3]);

        mask.x = (float)mask_bool[0];
        mask.y = (float)mask_bool[1];
        mask.z = (float)mask_bool[2];
        mask.w = (float)mask_bool[3];

        m_renderer.set_debug_color_mask(mask);

        for (uint32_t i = 0; i < m_forward_graph->node_count(); i++)
        {
            auto node = m_forward_graph->node(i);

            if (ImGui::TreeNode(node->name().c_str()))
            {
                auto& rts = node->output_render_targets();

                for (auto& output : rts)
                {
                    if (ImGui::Selectable(output.slot_name.c_str(), m_renderer.debug_render_target() == output.render_target))
                    {
                        if (m_renderer.debug_render_target() == output.render_target)
                            m_renderer.set_debug_render_target(nullptr);
                        else
                            m_renderer.set_debug_render_target(output.render_target);
                    }
                }

                for (int j = 0; j < node->intermediate_render_target_count(); j++)
                {
                    std::string name = node->name() + "_intermediate_" + std::to_string(j);

                    if (ImGui::Selectable(name.c_str(), m_renderer.debug_render_target() == node->intermediate_render_target(j)))
                    {
                        if (m_renderer.debug_render_target() == node->intermediate_render_target(j))
                            m_renderer.set_debug_render_target(nullptr);
                        else
                            m_renderer.set_debug_render_target(node->intermediate_render_target(j));
                    }
                }

                ImGui::TreePop();
            }
        }
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void edit_transform(const float* cameraView, float* cameraProjection, Transform* t, bool show_translate = true, bool show_rotate = true, bool show_scale = true)
    {
        static ImGuizmo::OPERATION mCurrentGizmoOperation(ImGuizmo::TRANSLATE);
        static ImGuizmo::MODE      mCurrentGizmoMode(ImGuizmo::WORLD);
        static bool                useSnap = false;
        static float               snap[3] = { 1.f, 1.f, 1.f };

        if (show_translate)
        {
            if (ImGui::RadioButton("Translate", mCurrentGizmoOperation == ImGuizmo::TRANSLATE))
                mCurrentGizmoOperation = ImGuizmo::TRANSLATE;
        }
        if (show_rotate)
        {
            ImGui::SameLine();
            if (ImGui::RadioButton("Rotate", mCurrentGizmoOperation == ImGuizmo::ROTATE))
                mCurrentGizmoOperation = ImGuizmo::ROTATE;
        }
        if (show_scale)
        {
            ImGui::SameLine();
            if (ImGui::RadioButton("Scale", mCurrentGizmoOperation == ImGuizmo::SCALE))
                mCurrentGizmoOperation = ImGuizmo::SCALE;
        }

        glm::vec3 position;
        glm::vec3 rotation;
        glm::vec3 scale;

        ImGuizmo::DecomposeMatrixToComponents(&t->model[0][0], &position.x, &rotation.x, &scale.x);

        if (show_translate)
            ImGui::InputFloat3("Tr", &position.x, 3);
        if (show_rotate)
            ImGui::InputFloat3("Rt", &rotation.x, 3);
        if (show_scale)
            ImGui::InputFloat3("Sc", &scale.x, 3);

        ImGuizmo::RecomposeMatrixFromComponents(&position.x, &rotation.x, &scale.x, (float*)&t->model);

        if (mCurrentGizmoOperation != ImGuizmo::SCALE)
        {
            if (ImGui::RadioButton("Local", mCurrentGizmoMode == ImGuizmo::LOCAL))
                mCurrentGizmoMode = ImGuizmo::LOCAL;
            ImGui::SameLine();
            if (ImGui::RadioButton("World", mCurrentGizmoMode == ImGuizmo::WORLD))
                mCurrentGizmoMode = ImGuizmo::WORLD;
        }
        ImGui::Checkbox("", &useSnap);
        ImGui::SameLine();

        switch (mCurrentGizmoOperation)
        {
            case ImGuizmo::TRANSLATE:
                ImGui::InputFloat3("Snap", &snap[0]);
                break;
            case ImGuizmo::ROTATE:
                ImGui::InputFloat("Angle Snap", &snap[0]);
                break;
            case ImGuizmo::SCALE:
                ImGui::InputFloat("Scale Snap", &snap[0]);
                break;
        }

        ImGuiIO& io = ImGui::GetIO();
        ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);
        ImGuizmo::Manipulate(cameraView, cameraProjection, mCurrentGizmoOperation, mCurrentGizmoMode, (float*)&t->model, NULL, useSnap ? &snap[0] : NULL);

        glm::vec3 temp;
        ImGuizmo::DecomposeMatrixToComponents((float*)&t->model, &t->position.x, &temp.x, &t->scale.x);
        t->prev_model  = t->model;
        t->orientation = glm::quat(glm::radians(temp));
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    bool load_scene_from_dialog()
    {
        std::shared_ptr<Scene> scene      = nullptr;
        nfdchar_t*             scene_path = nullptr;
        nfdresult_t            result     = NFD_OpenDialog("json", nullptr, &scene_path);

        if (result == NFD_OKAY)
        {
            scene = m_resource_manager.load_scene(scene_path, true);
            free(scene_path);

            if (!scene)
            {
                NIMBLE_LOG_ERROR("Failed to load scene!");
                return false;
            }
            else
            {
                if (m_scene)
                    m_renderer.shader_cache().clear_generated_cache();

				m_scene = scene;
            }

            return true;
        }
        else if (result == NFD_CANCEL)
            return false;
        else
        {
            std::string error = "Scene file read error: ";
            error += NFD_GetError();
            NIMBLE_LOG_ERROR(error);
            return false;
        }
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void update_camera()
    {
        if (m_scene)
        {
            auto current = m_scene->camera();

            float forward_delta = m_heading_speed * m_delta;
            float right_delta   = m_sideways_speed * m_delta;

            current->set_translation_delta(current->m_forward, forward_delta);
            current->set_translation_delta(current->m_right, right_delta);

            if (m_mouse_look)
            {
                // Activate Mouse Look
                current->set_rotatation_delta(glm::vec3((float)(m_mouse_delta_y * m_camera_sensitivity),
                                                        (float)(m_mouse_delta_x * m_camera_sensitivity),
                                                        (float)(0.0f)));
            }
            else
            {
                current->set_rotatation_delta(glm::vec3((float)(0),
                                                        (float)(0),
                                                        (float)(0)));
            }

            current->update();
        }
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

private:
    // Camera controls.
    bool  m_mouse_look         = false;
    bool  m_edit_mode          = true;
    bool  m_debug_mode         = false;
    bool  m_debug_gui          = false;
    bool  m_move_entities      = false;
    float m_heading_speed      = 0.0f;
    float m_sideways_speed     = 0.0f;
    float m_camera_sensitivity = 0.05f;
    float m_camera_speed       = 0.1f;

    std::shared_ptr<Scene>                 m_scene;
    std::shared_ptr<Viewport>              m_viewport;
    std::shared_ptr<RenderGraph>           m_forward_graph;
    std::shared_ptr<RenderGraph>           m_pcf_point_light_graph;
    std::shared_ptr<RenderGraph>           m_pcf_spot_light_graph;
    std::shared_ptr<RenderGraph>           m_pcf_directional_light_graph;
    std::shared_ptr<BrunetonProbeRenderer> m_bruneton_probe_renderer;

    Entity::ID           m_selected_entity      = UINT32_MAX;
    PointLight::ID       m_selected_point_light = UINT32_MAX;
    SpotLight::ID        m_selected_spot_light  = UINT32_MAX;
    DirectionalLight::ID m_selected_dir_light   = UINT32_MAX;
};
} // namespace nimble

NIMBLE_DECLARE_MAIN(nimble::Nimble)