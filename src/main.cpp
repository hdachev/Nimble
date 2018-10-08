#include <iostream>
#include <fstream>
#include <gtc/matrix_transform.hpp>
#include <gtc/type_ptr.hpp>
#include <application.h>
#include <camera.h>
#include <utility.h>
#include <material.h>
#include <macros.h>
#include <memory>
#include <debug_draw.h>
#include <imgui_helpers.h>

#include "external/nfd/nfd.h"
#include "renderer.h"

#define CAMERA_FAR_PLANE 1000.0f

class Nimble : public dw::Application
{
protected:
	// -----------------------------------------------------------------------------------------------------------------------------------

    bool init(int argc, const char* argv[]) override
    {
		m_renderer = std::make_unique<Renderer>();

		// Attempt to load startup scene.
		Scene* scene = Scene::load(dw::utility::path_for_resource("/assets/scene/startup.json"));

		// If failed, prompt user to select scene to be loaded.
		if (!scene)
		{
			nfdchar_t* scene_path = nullptr;
			nfdresult_t result = NFD_OpenDialog("json", nullptr, &scene_path);

			if (result == NFD_OKAY)
			{
				scene = Scene::load(scene_path);
				free(scene_path);
			}
			else if (result == NFD_CANCEL)
				return false;
			else
			{
				std::string error = "Scene file read error: ";
				error += NFD_GetError();
				DW_LOG_ERROR(error);
				return false;
			}
		}

		if (!scene)
		{
			DW_LOG_ERROR("Failed to load scene!");
			return false;
		}
		else
			m_scene = std::unique_ptr<Scene>(scene);

		// Create camera.
		create_camera();

		m_renderer->initialize(m_width, m_height, m_main_camera.get());

		m_renderer->set_scene(m_scene.get());

		return true;
    }

	// -----------------------------------------------------------------------------------------------------------------------------------

    void update(double delta) override
    {
		// Update camera.
		update_camera();

		m_scene->update();

		if (m_debug_gui)
			m_renderer->debug_gui(delta);

		m_renderer->render(delta);
    }

	// -----------------------------------------------------------------------------------------------------------------------------------

    void shutdown() override
    {

    }

	// -----------------------------------------------------------------------------------------------------------------------------------

	dw::AppSettings intial_app_settings() override
	{
		dw::AppSettings settings;
			
		settings.resizable = true;
		settings.width = 1280;
		settings.height = 720;
		settings.title = "Nimble - Dihara Wijetunga (c) 2018";

		return settings;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void window_resized(int width, int height) override
	{
		m_main_camera->m_width = m_width;
		m_main_camera->m_height = m_height;

		// Override window resized method to update camera projection.
		m_main_camera->update_projection(60.0f, 0.1f, CAMERA_FAR_PLANE, float(m_width) / float(m_height));
		m_debug_camera->update_projection(60.0f, 0.1f, CAMERA_FAR_PLANE * 2.0f, float(m_width) / float(m_height));

		m_renderer->on_window_resized(width, height);
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
		m_main_camera = std::make_unique<dw::Camera>(60.0f, 0.1f, CAMERA_FAR_PLANE, float(m_width) / float(m_height), glm::vec3(0.0f, 5.0f, 20.0f), glm::vec3(0.0f, 0.0, -1.0f));
		m_debug_camera = std::make_unique<dw::Camera>(60.0f, 0.1f, CAMERA_FAR_PLANE * 2.0f, float(m_width) / float(m_height), glm::vec3(0.0f, 5.0f, 20.0f), glm::vec3(0.0f, 0.0, -1.0f));

		m_main_camera->m_width = m_width;
		m_main_camera->m_height = m_height;
		m_main_camera->m_half_pixel_jitter = true;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void update_camera()
	{
		dw::Camera* current = m_main_camera.get();

		if (m_debug_mode)
			current = m_debug_camera.get();

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
	// Camera.
	std::unique_ptr<dw::Camera> m_main_camera;
	std::unique_ptr<dw::Camera> m_debug_camera;

	// Camera controls.
	bool m_mouse_look = false;
	bool m_debug_mode = false;
	bool m_debug_gui = false;
	bool m_move_entities = false;
	float m_heading_speed = 0.0f;
	float m_sideways_speed = 0.0f;
	float m_camera_sensitivity = 0.05f;
	float m_camera_speed = 0.1f;

	std::unique_ptr<Renderer> m_renderer;
	std::unique_ptr<Scene> m_scene;
};

DW_DECLARE_MAIN(Nimble)
