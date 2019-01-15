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
#include "profiler.h"
#include <random>

namespace nimble
{
	#define CAMERA_FAR_PLANE 10000.0f

	class Nimble : public Application
	{
	protected:
		// -----------------------------------------------------------------------------------------------------------------------------------

		bool init(int argc, const char* argv[]) override
		{
			// Attempt to load startup scene.
			std::shared_ptr<Scene> scene = m_resource_manager.load_scene("scene/startup.json");

			// If failed, prompt user to select scene to be loaded.
			if (!scene)
			{
				nfdchar_t* scene_path = nullptr;
				nfdresult_t result = NFD_OpenDialog("json", nullptr, &scene_path);

				if (result == NFD_OKAY)
				{
					scene = m_resource_manager.load_scene(scene_path);
					free(scene_path);
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

			if (!scene)
			{
				NIMBLE_LOG_ERROR("Failed to load scene!");
				return false;
			}
			else
				m_scene = scene;

			//m_scene->create_point_light(glm::vec3(0.0f, 100.0f, 0.0f), glm::vec3(1.0f), 2000.0f, 10.0f, false);

			m_scene->create_spot_light(glm::vec3(0.0f, 100.0f, 0.0f), glm::vec3(90.0f, 0.0f, 0.0f), glm::vec3(1.0f), 45.0f, 1000.0f, 10.0f);
			//create_random_point_lights();

			// Create camera.
			create_camera();

			m_forward_graph = std::make_shared<ForwardRenderGraph>(&m_renderer);

			m_renderer.register_render_graph(m_forward_graph);
			m_renderer.set_scene(m_scene);
			m_renderer.set_scene_render_graph(m_forward_graph);

			return true;
		}

		// -----------------------------------------------------------------------------------------------------------------------------------

		void update(double delta) override
		{
			// Update camera.
			update_camera();

			gui();

			m_scene->update();

			m_renderer.render();
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
			m_scene->camera()->m_width = m_width;
			m_scene->camera()->m_height = m_height;

			// Override window resized method to update camera projection.
			m_scene->camera()->update_projection(60.0f, 0.1f, CAMERA_FAR_PLANE, float(m_width) / float(m_height));

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
			if (code == GLFW_MOUSE_BUTTON_LEFT)
				m_mouse_look = true;
		}

		// -----------------------------------------------------------------------------------------------------------------------------------

		void mouse_released(int code) override
		{
			// Disable mouse look.
			if (code == GLFW_MOUSE_BUTTON_LEFT)
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
			{
				glm::vec3 position = glm::vec3(dis_x(rd), dis_y(rd), dis_z(rd));
				glm::vec3 color = glm::vec3(dis(rd), dis(rd), dis(rd));

				m_scene->create_point_light(position, color, range, intensity, false);
			}
		}

		// -----------------------------------------------------------------------------------------------------------------------------------

		void gui()
		{
			float cpu_time = 0.0f;
			float gpu_time = 0.0f;

			Profiler::cpu_result(PROFILER_FRUSTUM_CULLING, cpu_time);

			ImGui::Text("Frustum Culling: %f(CPU), 0.0(GPU)", cpu_time);

			for (uint32_t i = 0; i < m_forward_graph->node_count(); i++)
			{
				std::shared_ptr<RenderNode> node = m_forward_graph->node(i);

				node->timing_total(cpu_time, gpu_time);

				ImGui::Text("%s: %f(CPU), %f(GPU)", node->name().c_str(), cpu_time, gpu_time);
			}
		}

		// -----------------------------------------------------------------------------------------------------------------------------------

		void update_camera()
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

		// -----------------------------------------------------------------------------------------------------------------------------------

	private:
		// Camera controls.
		bool m_mouse_look = false;
		bool m_debug_mode = false;
		bool m_debug_gui = false;
		bool m_move_entities = false;
		float m_heading_speed = 0.0f;
		float m_sideways_speed = 0.0f;
		float m_camera_sensitivity = 0.05f;
		float m_camera_speed = 0.1f;

		std::shared_ptr<Scene> m_scene;
		std::shared_ptr<ForwardRenderGraph> m_forward_graph;
	};
}

NIMBLE_DECLARE_MAIN(nimble::Nimble)