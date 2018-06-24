#include "renderer.h"
#include <camera.h>
#include <material.h>
#include <mesh.h>
#include <logger.h>
#include <utility.h>
#include <fstream>
#include <imgui.h>

#include "entity.h"
#include "global_graphics_resources.h"
#include "constants.h"

// -----------------------------------------------------------------------------------------------------------------------------------

Renderer::Renderer() {}

// -----------------------------------------------------------------------------------------------------------------------------------

Renderer::~Renderer() {}

void Renderer::initialize(uint16_t width, uint16_t height, dw::Camera* camera)
{
	m_width = width;
	m_height = height;
	m_current_output = 0;
	set_camera(camera);

	glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

	// Initialize global resource.
	GlobalGraphicsResources::initialize();

	//// Load cubemap shaders
	//{
	//	std::string path = "shader/cubemap_vs.glsl";
	//	m_cube_map_vs = load_shader(GL_VERTEX_SHADER, path, nullptr);
	//	path = "shader/cubemap_fs.glsl";
	//	m_cube_map_fs = load_shader(GL_FRAGMENT_SHADER, path, nullptr);

	//	dw::Shader* shaders[] = { m_cube_map_vs, m_cube_map_fs };

	//	path = dw::utility::executable_path() + "/cubemap_vs.glslcubemap_fs.glsl";
	//	m_cube_map_program = load_program(path, 2, &shaders[0]);

	//	if (!m_cube_map_vs || !m_cube_map_fs || !m_cube_map_program)
	//	{
	//		DW_LOG_ERROR("Failed to load cubemap shaders");
	//	}
	//}

	//// Load shadowmap shaders
	//{
	//	std::string path = "shader/pssm_vs.glsl";
	//	m_pssm_vs = load_shader(GL_VERTEX_SHADER, path, nullptr);
	//	path = "shader/pssm_fs.glsl";
	//	m_pssm_fs = load_shader(GL_FRAGMENT_SHADER, path, nullptr);

	//	dw::Shader* shaders[] = { m_pssm_vs, m_pssm_fs };

	//	path = dw::utility::executable_path() + "/pssm_vs.glslpssm_fs.glsl";
	//	m_pssm_program = load_program(path, 2, &shaders[0]);

	//	if (!m_pssm_vs || !m_pssm_fs || !m_pssm_program)
	//	{
	//		DW_LOG_ERROR("Failed to load PSSM shaders");
	//	}
	//}

	m_per_scene_uniforms.pointLightCount = 0;
	m_per_scene_uniforms.pointLights[0].position = glm::vec4(-10.0f, 20.0f, 10.0f, 1.0f);
	m_per_scene_uniforms.pointLights[0].color = glm::vec4(300.0f);
	m_per_scene_uniforms.pointLights[1].position = glm::vec4(10.0f, 20.0f, 10.0f, 1.0f);
	m_per_scene_uniforms.pointLights[1].color = glm::vec4(300.0f);
	m_per_scene_uniforms.pointLights[2].position = glm::vec4(-10.0f, -20.0f, 10.0f, 1.0f);
	m_per_scene_uniforms.pointLights[2].color = glm::vec4(300.0f);
	m_per_scene_uniforms.pointLights[3].position = glm::vec4(10.0f, -20.0f, 10.0f, 1.0f);
	m_per_scene_uniforms.pointLights[3].color = glm::vec4(300.0f);

	m_light_direction = glm::vec3(0.0f, -1.0f, 0.0f);

	m_per_scene_uniforms.directionalLight.color = glm::vec4(1.0f, 1.0f, 1.0f, 20.0f);
	m_per_scene_uniforms.directionalLight.direction = glm::vec4(m_light_direction, 1.0f);

	// Initialize renderers
	m_forward_renderer.initialize(m_width, m_height);
	m_gbuffer_renderer.initialize(m_width, m_height);

	// Initialize CSM.
	m_csm_technique.initialize(0.3, 350.0f, 4, 2048, m_camera, m_width, m_height, m_light_direction);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Renderer::shutdown()
{
	// Shutdown CSM.
	m_csm_technique.shutdown();

	// Shutdown renderers.
	m_forward_renderer.shutdown();

	// Clean up global resources.
	GlobalGraphicsResources::shutdown();
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Renderer::set_scene(Scene* scene)
{
	m_scene = scene;
}

// -----------------------------------------------------------------------------------------------------------------------------------


void Renderer::set_camera(dw::Camera* camera)
{
	m_camera = camera;
}

// -----------------------------------------------------------------------------------------------------------------------------------

Scene* Renderer::scene()
{
	return m_scene;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Renderer::on_window_resized(uint16_t width, uint16_t height)
{
	m_width = width;
	m_height = height;

	// Propagate window resize to renderers.
	m_forward_renderer.on_window_resized(width, height);
	m_gbuffer_renderer.on_window_resized(width, height);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Renderer::update_uniforms(dw::Camera* camera)
{
	Entity** entities = m_scene->entities();
	int entity_count = m_scene->entity_count();

	m_per_frame_uniforms.projMat = camera->m_projection;
	m_per_frame_uniforms.viewMat = camera->m_view;
	m_per_frame_uniforms.viewProj = camera->m_view_projection;
	m_per_frame_uniforms.viewDir = glm::vec4(camera->m_forward.x, camera->m_forward.y, camera->m_forward.z, 0.0f);
	m_per_frame_uniforms.viewPos = glm::vec4(camera->m_position.x, camera->m_position.y, camera->m_position.z, 0.0f);
	m_per_frame_uniforms.numCascades = m_csm_technique.frustum_split_count();
	
	m_per_scene_uniforms.directionalLight.direction = glm::vec4(glm::normalize(m_light_direction), 1.0f);

	for (int i = 0; i < m_per_frame_uniforms.numCascades; i++)
	{
		m_per_frame_uniforms.shadowFrustums[i].farPlane = m_csm_technique.far_bound(i);
		m_per_frame_uniforms.shadowFrustums[i].shadowMatrix = m_csm_technique.texture_matrix(i);
	}

	for (int i = 0; i < entity_count; i++)
	{
		Entity* entity = entities[i];
		m_per_entity_uniforms[i].modalMat = entity->m_transform;
		m_per_entity_uniforms[i].mvpMat = camera->m_view_projection * entity->m_transform;
		m_per_entity_uniforms[i].worldPos = glm::vec4(entity->m_position.x, entity->m_position.y, entity->m_position.z, 0.0f);
	}

	void* mem = GlobalGraphicsResources::per_frame_ubo()->map(GL_WRITE_ONLY);

	if (mem)
	{
		memcpy(mem, &m_per_frame_uniforms, sizeof(PerFrameUniforms));
		GlobalGraphicsResources::per_frame_ubo()->unmap();
	}

	mem = GlobalGraphicsResources::per_scene_ubo()->map(GL_WRITE_ONLY);

	if (mem)
	{
		memcpy(mem, &m_per_scene_uniforms, sizeof(PerSceneUniforms));
		GlobalGraphicsResources::per_scene_ubo()->unmap();
	}

	mem = GlobalGraphicsResources::per_entity_ubo()->map(GL_WRITE_ONLY);

	if (mem)
	{
		memcpy(mem, &m_per_entity_uniforms[0], sizeof(PerEntityUniforms) * entity_count);
		GlobalGraphicsResources::per_entity_ubo()->unmap();
	}
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Renderer::debug_gui(double delta)
{
	if (ImGui::Begin("Nimble - Debug GUI"))
	{
		ImGui::Text("Frame Time: %f ms", delta);

		ImGui::Text("Current Output:");

		ImGui::RadioButton("Scene", &m_current_output, SHOW_COLOR);
		ImGui::RadioButton("Forward Depth (Linear)", &m_current_output, SHOW_FORWARD_DEPTH);
		ImGui::RadioButton("G-Buffer - Albedo", &m_current_output, SHOW_GBUFFER_ALBEDO);
		ImGui::RadioButton("G-Buffer - Normals", &m_current_output, SHOW_GBUFFER_NORMALS);
		ImGui::RadioButton("G-Buffer - Roughness", &m_current_output, SHOW_GBUFFER_ROUGHNESS);
		ImGui::RadioButton("G-Buffer - Metalness", &m_current_output, SHOW_GBUFFER_METALNESS);
		ImGui::RadioButton("G-Buffer - Velocity", &m_current_output, SHOW_GBUFFER_VELOCITY);

		for (int i = 0; i < m_csm_technique.m_split_count; i++)
		{
			std::string name = "Cascade " + std::to_string(i + 1);
			ImGui::RadioButton(name.c_str(), &m_current_output, SHOW_SHADOW_MAPS + i);
		}

		ImGui::SliderFloat("Light Direction X", &m_light_direction.x, -1.0f, 1.0f);
		ImGui::SliderFloat("Light Direction Z", &m_light_direction.z, -1.0f, 1.0f);
	}
	ImGui::End();
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Renderer::render()
{
	// Check if scene has been set.
	if (!m_scene)
	{
		DW_LOG_ERROR("Scene has not been set!");
		return;
	}

	// Update CSM.
	m_csm_technique.update(m_camera, glm::normalize(m_light_direction));

	// Update per-frame and per-entity uniforms.
	update_uniforms(m_camera);

	// Dispatch shadow map rendering.
	m_shadow_map_renderer.render(m_scene, &m_csm_technique);

	// Dispatch scene rendering.
	m_forward_renderer.render(m_scene, m_width, m_height);
	m_gbuffer_renderer.render(m_scene, m_width, m_height);

	// Render final composition.
	m_final_composition.render(m_camera, m_width, m_height, m_current_output);
}

// -----------------------------------------------------------------------------------------------------------------------------------