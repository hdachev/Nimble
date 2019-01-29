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
#include "debug_draw.h"
#include "imgui_helpers.h"
#include "external/nfd/nfd.h"
#include "graphs/forward_render_graph.h"
#include "graphs/pcf_point_light_render_graph.h"
#include "graphs/pcf_spot_light_render_graph.h"
#include "profiler.h"
#include "ImGuizmo.h"
#include <random>

#define NIMBLE_EDITOR

namespace nimble
{
	#define CAMERA_FAR_PLANE 5000.0f

	class Nimble : public Application
	{
	protected:
		// -----------------------------------------------------------------------------------------------------------------------------------

		bool init(int argc, const char* argv[]) override
		{
			// Attempt to load startup scene.
			std::shared_ptr<Scene> scene = m_resource_manager.load_scene("scene/startup.json");

			// If failed, prompt user to select scene to be loaded.
			if (!scene && !load_scene_from_dialog())
				return false;
			else
				m_scene = scene;

			m_scene->create_directional_light(glm::vec3(45.0f, 0.0f, 0.0f), glm::vec3(1.0f), 10.0f);
			//create_random_point_lights();
			//create_random_spot_lights();

			// Create camera.
			create_camera();

			m_forward_graph = std::make_shared<ForwardRenderGraph>(&m_renderer);
			m_pcf_point_light_graph = std::make_shared<PCFPointLightRenderGraph>(&m_renderer);
			m_pcf_spot_light_graph = std::make_shared<PCFSpotLightRenderGraph>(&m_renderer);

			m_renderer.register_render_graph(m_forward_graph);
			m_renderer.register_render_graph(m_pcf_point_light_graph);
			m_renderer.register_render_graph(m_pcf_spot_light_graph);

			m_renderer.set_scene(m_scene);
			m_renderer.set_scene_render_graph(m_forward_graph);
			m_renderer.set_point_light_render_graph(m_pcf_point_light_graph);
			m_renderer.set_spot_light_render_graph(m_pcf_spot_light_graph);

			return true;
		}

		// -----------------------------------------------------------------------------------------------------------------------------------

		void update(double delta) override
		{
			// Update camera.
			update_camera();

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
			m_renderer.render();

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
			settings.width = 1280;
			settings.height = 720;
			settings.title = "Nimble - Dihara Wijetunga (c) 2018";

			return settings;
		}

		// -----------------------------------------------------------------------------------------------------------------------------------

		void window_resized(int width, int height) override
		{
			if (m_scene)
			{
				m_scene->camera()->m_width = m_width;
				m_scene->camera()->m_height = m_height;

				// Override window resized method to update camera projection.
				m_scene->camera()->update_projection(60.0f, 0.1f, CAMERA_FAR_PLANE, float(m_width) / float(m_height));
			}

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
			m_scene->camera()->m_width = m_width;
			m_scene->camera()->m_height = m_height;
			m_scene->camera()->m_half_pixel_jitter = false;
			m_scene->camera()->update_projection(60.0f, 0.1f, CAMERA_FAR_PLANE, float(m_width) / float(m_height));
		}

		// -----------------------------------------------------------------------------------------------------------------------------------

		void create_random_spot_lights()
		{
			AABB aabb = m_scene->aabb();

			const float range = 300.0f;
			const float intensity = 10.0f;
			const uint32_t num_lights = 10;
			const float aabb_scale = 0.6f;

			std::random_device rd;
			std::mt19937 gen(rd());

			std::uniform_real_distribution<float> dis(0.0f, 1.0f);
			std::uniform_real_distribution<float> dis_x(aabb.min.x * aabb_scale, aabb.max.x * aabb_scale);
			std::uniform_real_distribution<float> dis_y(aabb.min.y * aabb_scale, aabb.max.y * aabb_scale);
			std::uniform_real_distribution<float> dis_z(aabb.min.z * aabb_scale, aabb.max.z * aabb_scale);
			std::uniform_real_distribution<float> dis_pitch(0.0f, 180.0f);
			std::uniform_real_distribution<float> dis_yaw(0.0f, 360.0f);

			for (int n = 0; n < num_lights; n++)
				m_scene->create_spot_light(glm::vec3(dis_x(rd), dis_y(rd), dis_z(rd)), glm::vec3(dis_pitch(rd), dis_yaw(rd), 0.0f), glm::vec3(dis(rd), dis(rd), dis(rd)), 45.0f, 1000.0f, 10.0f);
		}

		// -----------------------------------------------------------------------------------------------------------------------------------

		void create_random_point_lights()
		{
			AABB aabb = m_scene->aabb();

			const float range = 300.0f;
			const float intensity = 10.0f;
			const uint32_t num_lights = 100;
			const float aabb_scale = 0.6f;

			std::random_device rd;
			std::mt19937 gen(rd());

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
						ImGui::InputFloat("Intensity", &light.intensity);
						ImGui::InputFloat("Range", &light.range);
						ImGui::InputFloat("Cone Angle", &light.cone_angle);
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
				{
					Profiler::cpu_result(PROFILER_FRUSTUM_CULLING, cpu_time);

					ImGui::Text("Frustum Culling: %f(CPU), 0.0(GPU)", cpu_time);

					for (uint32_t i = 0; i < m_forward_graph->node_count(); i++)
					{
						std::shared_ptr<RenderNode> node = m_forward_graph->node(i);

						node->timing_total(cpu_time, gpu_time);

						ImGui::Text("%s: %f(CPU), %f(GPU)", node->name().c_str(), cpu_time, gpu_time);
					}

					for (uint32_t i = 0; i < m_pcf_point_light_graph->node_count(); i++)
					{
						std::shared_ptr<RenderNode> node = m_pcf_point_light_graph->node(i);

						node->timing_total(cpu_time, gpu_time);

						ImGui::Text("%s: %f(CPU), %f(GPU)", node->name().c_str(), cpu_time, gpu_time);
					}
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
								m_selected_entity = entities[i].id;
								m_selected_dir_light = UINT32_MAX;
								m_selected_point_light = UINT32_MAX;
								m_selected_spot_light = UINT32_MAX;
							}
						}
					}
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
								m_selected_entity = UINT32_MAX;
								m_selected_dir_light = UINT32_MAX;
								m_selected_point_light = lights[i].id;
								m_selected_spot_light = UINT32_MAX;
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
								m_selected_entity = UINT32_MAX;
								m_selected_dir_light = UINT32_MAX;
								m_selected_point_light = UINT32_MAX;
								m_selected_spot_light = lights[i].id;
							}
						}

						ImGui::Separator();

						ImGui::PushID(2);
						if (ImGui::Button("Create"))
							m_scene->create_spot_light(glm::vec3(0.0f), glm::vec3(0.0f), glm::vec3(1.0f), 45.0f, 100.0f, 1.0f);

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
								m_selected_entity = UINT32_MAX;
								m_selected_dir_light = lights[i].id;
								m_selected_point_light = UINT32_MAX;
								m_selected_spot_light = UINT32_MAX;
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

		void edit_transform(const float* cameraView, float* cameraProjection, Transform* t, bool show_translate = true, bool show_rotate = true, bool show_scale = true)
		{
			static ImGuizmo::OPERATION mCurrentGizmoOperation(ImGuizmo::TRANSLATE);
			static ImGuizmo::MODE mCurrentGizmoMode(ImGuizmo::WORLD);
			static bool useSnap = false;
			static float snap[3] = { 1.f, 1.f, 1.f };

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
			t->orientation = glm::quat(glm::radians(temp));
		}

		// -----------------------------------------------------------------------------------------------------------------------------------

		bool load_scene_from_dialog()
		{
			std::shared_ptr<Scene> scene = nullptr;
			nfdchar_t* scene_path = nullptr;
			nfdresult_t result = NFD_OpenDialog("json", nullptr, &scene_path);

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
					m_scene = scene;

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
				float right_delta = m_sideways_speed * m_delta;

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
		bool m_mouse_look = false;
		bool m_edit_mode = true;
		bool m_debug_mode = false;
		bool m_debug_gui = false;
		bool m_move_entities = false;
		float m_heading_speed = 0.0f;
		float m_sideways_speed = 0.0f;
		float m_camera_sensitivity = 0.05f;
		float m_camera_speed = 0.1f;

		std::shared_ptr<Scene> m_scene;
		std::shared_ptr<ForwardRenderGraph> m_forward_graph;
		std::shared_ptr<PCFPointLightRenderGraph> m_pcf_point_light_graph;
		std::shared_ptr<PCFSpotLightRenderGraph> m_pcf_spot_light_graph;

		Entity::ID m_selected_entity = UINT32_MAX;
		PointLight::ID m_selected_point_light = UINT32_MAX;
		SpotLight::ID m_selected_spot_light = UINT32_MAX;
		DirectionalLight::ID m_selected_dir_light = UINT32_MAX;
	};
}

NIMBLE_DECLARE_MAIN(nimble::Nimble)