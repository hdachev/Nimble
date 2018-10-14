#include "renderer.h"
#include <camera.h>
#include <material.h>
#include <mesh.h>
#include <logger.h>
#include <utility.h>
#include <fstream>
#include <imgui.h>
#include <gtc/matrix_transform.hpp>

#include "entity.h"
#include "global_graphics_resources.h"
#include "constants.h"
#include "gpu_profiler.h"

static const char* g_renderer_names[] = 
{
	"Forward",
	"Deferred"
};

// -----------------------------------------------------------------------------------------------------------------------------------

Renderer::Renderer() {}

// -----------------------------------------------------------------------------------------------------------------------------------

Renderer::~Renderer() {}

void Renderer::initialize(uint16_t width, uint16_t height, dw::Camera* camera)
{
	m_width = width;
	m_height = height;
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
	m_deferred_shading_renderer.initialize(m_width, m_height);

	// Initialize effects
	m_motion_blur.initialize(m_width, m_height);
	m_ssr.initialize(m_width, m_height);
	m_bloom.initialize(m_width, m_height);
	m_tone_mapping.initialize(m_width, m_height);
	m_ambient_occlusion.initialize(m_width, m_height);
	m_taa.initialize(m_width, m_height);
	m_depth_of_field.initialize(m_width, m_height);

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
	m_gbuffer_renderer.shutdown();
	m_ambient_occlusion.shutdown();
	m_motion_blur.shutdown();
	m_ssr.shutdown();
	m_bloom.shutdown();
	m_tone_mapping.shutdown();
	m_taa.shutdown();
	m_depth_of_field.shutdown();
	m_deferred_shading_renderer.shutdown();

	// Clean up profiler queries
	GPUProfiler::shutdown();

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
	m_deferred_shading_renderer.on_window_resized(width, height);
	m_motion_blur.on_window_resized(width, height);
	m_ssr.on_window_resized(width, height);
	m_bloom.on_window_resized(width, height);
	m_tone_mapping.on_window_resized(width, height);
	m_ambient_occlusion.on_window_resized(width, height);
	m_taa.on_window_resized(width, height);
	m_depth_of_field.on_window_resized(width, height);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Renderer::update_uniforms(dw::Camera* camera, double delta)
{
	Entity** entities = m_scene->entities();
	int entity_count = m_scene->entity_count();

	PerFrameUniforms& per_frame = GlobalGraphicsResources::per_frame_uniforms();

	per_frame.lastViewProj = camera->m_prev_view_projection;
	per_frame.projMat = camera->m_projection;
	per_frame.viewMat = camera->m_view;
	per_frame.viewProj = camera->m_view_projection;
	per_frame.invViewProj = glm::inverse(camera->m_view_projection);
	per_frame.invProj = glm::inverse(camera->m_projection);
	per_frame.invView = glm::inverse(camera->m_view);
	per_frame.viewDir = glm::vec4(camera->m_forward.x, camera->m_forward.y, camera->m_forward.z, 0.0f);
	per_frame.viewPos = glm::vec4(camera->m_position.x, camera->m_position.y, camera->m_position.z, 0.0f);
	per_frame.current_prev_jitter = glm::vec4(camera->m_current_jitter, camera->m_prev_jitter);
	per_frame.numCascades = m_csm_technique.frustum_split_count();
	per_frame.tanHalfFov = glm::tan(glm::radians(camera->m_fov / 2.0f));
	per_frame.aspectRatio = float(m_width) / float(m_height);
	per_frame.farPlane = camera->m_far;
	per_frame.nearPlane = camera->m_near;
	per_frame.viewport_width = m_width;
	per_frame.viewport_height = m_height;

	int current_fps = int((1.0f / float(delta)) * 1000.0f);
	int target_fps = 60;

	per_frame.velocity_scale = float(current_fps) / float(target_fps);
	
	for (int i = 0; i < per_frame.numCascades; i++)
	{
		per_frame.shadowFrustums[i].farPlane = m_csm_technique.far_bound(i);
		per_frame.shadowFrustums[i].shadowMatrix = m_csm_technique.texture_matrix(i);
	}

	m_per_scene_uniforms.directionalLight.direction = glm::vec4(glm::normalize(m_light_direction), 1.0f);

	for (int i = 0; i < entity_count; i++)
	{
		Entity* entity = entities[i];
		m_per_entity_uniforms[i].modalMat = entity->m_transform;
		m_per_entity_uniforms[i].mvpMat = camera->m_view_projection * entity->m_transform;
		m_per_entity_uniforms[i].lastMvpMat = camera->m_prev_view_projection * entity->m_prev_transform;
		m_per_entity_uniforms[i].worldPos = glm::vec4(entity->m_position.x, entity->m_position.y, entity->m_position.z, 0.0f);
	}

	void* mem = GlobalGraphicsResources::per_frame_ubo()->map(GL_WRITE_ONLY);

	if (mem)
	{
		memcpy(mem, &per_frame, sizeof(PerFrameUniforms));
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

		PerFrameUniforms& per_frame = GlobalGraphicsResources::per_frame_uniforms();

		int index = per_frame.renderer;

		if (ImGui::BeginCombo("Renderer", g_renderer_names[index]))
		{
			for (int i = 0; i < 2; i++)
			{
				if (ImGui::Selectable(g_renderer_names[i], index == i))
				{
					per_frame.renderer = i;

					if (per_frame.renderer == RENDERER_FORWARD)
					{
						if (per_frame.current_output > SHOW_FORWARD_DEPTH && per_frame.current_output < SHOW_SHADOW_MAPS)
							per_frame.current_output = SHOW_FORWARD_COLOR;
					}
					else if (per_frame.renderer == RENDERER_DEFERRED)
					{
						if (per_frame.current_output < SHOW_DEFERRED_COLOR)
							per_frame.current_output = SHOW_DEFERRED_COLOR;
					}
				}
			}
			ImGui::EndCombo();
		}

		if (ImGui::CollapsingHeader("Current Output:"))
		{
			if (per_frame.renderer == RENDERER_FORWARD)
			{
				ImGui::RadioButton("Scene", &per_frame.current_output, SHOW_FORWARD_COLOR);
				ImGui::RadioButton("Forward Depth (Linear)", &per_frame.current_output, SHOW_FORWARD_DEPTH);
			}
			else if (per_frame.renderer == RENDERER_DEFERRED)
			{
				ImGui::RadioButton("Scene", &per_frame.current_output, SHOW_DEFERRED_COLOR);
				ImGui::RadioButton("G-Buffer - Albedo", &per_frame.current_output, SHOW_GBUFFER_ALBEDO);
				ImGui::RadioButton("G-Buffer - Normals", &per_frame.current_output, SHOW_GBUFFER_NORMALS);
				ImGui::RadioButton("G-Buffer - Roughness", &per_frame.current_output, SHOW_GBUFFER_ROUGHNESS);
				ImGui::RadioButton("G-Buffer - Metalness", &per_frame.current_output, SHOW_GBUFFER_METALNESS);
				ImGui::RadioButton("G-Buffer - Velocity", &per_frame.current_output, SHOW_GBUFFER_VELOCITY);
				ImGui::RadioButton("G-Buffer - Depth (Linear)", &per_frame.current_output, SHOW_GBUFFER_DEPTH);
				ImGui::RadioButton("SSR", &per_frame.current_output, SHOW_SSR);
			}

			ImGui::RadioButton("SSAO Buffer", &per_frame.current_output, SHOW_SSAO);
			ImGui::RadioButton("SSAO Blur", &per_frame.current_output, SHOW_SSAO_BLUR);
			ImGui::RadioButton("Bright Pass", &per_frame.current_output, SHOW_BRIGHT_PASS);

			for (int i = 0; i < m_csm_technique.m_split_count; i++)
			{
				std::string name = "Cascade " + std::to_string(i + 1);
				ImGui::RadioButton(name.c_str(), &per_frame.current_output, SHOW_SHADOW_MAPS + i);
			}
		}

		if (ImGui::CollapsingHeader("Directional Light"))
		{
			ImGui::SliderFloat("Light Direction X", &m_light_direction.x, -1.0f, 1.0f);
			ImGui::SliderFloat("Light Direction Z", &m_light_direction.z, -1.0f, 1.0f);
		}

		if (ImGui::CollapsingHeader("Camera"))
		{
			ImGui::SliderFloat("Near Field Begin", &m_depth_of_field.m_near_begin, 0.0f, 1.0f);
			ImGui::SliderFloat("Near Field End", &m_depth_of_field.m_near_end, 0.0f, 1.0f);
			ImGui::SliderFloat("Far Field Begin", &m_depth_of_field.m_far_begin, 0.0f, 1.0f);
			ImGui::SliderFloat("Far Field End", &m_depth_of_field.m_far_end, 0.0f, 1.0f);
		}

		if (ImGui::CollapsingHeader("Post-Process"))
		{
			ImGui::Checkbox("SSAO", (bool*)&per_frame.ssao);
			ImGui::Checkbox("Motion Blur", (bool*)&per_frame.motion_blur);

			bool bloom = m_bloom.is_enabled();
			ImGui::Checkbox("Bloom", &bloom);

			if (bloom)
				m_bloom.enable();
			else
				m_bloom.disable();

			bool ssr = m_ssr.is_enabled();
			ImGui::Checkbox("SSR", &ssr);

			if (ssr)
				m_ssr.enable();
			else
				m_ssr.disable();

			bool taa = m_taa.is_enabled();
			ImGui::Checkbox("TAA", &taa);

			if (taa)
			{
				m_camera->m_half_pixel_jitter = true;
				m_taa.enable();
			}
			else
			{
				m_camera->m_half_pixel_jitter = false;
				m_taa.disable();
			}

			int32_t current_operator = m_tone_mapping.current_operator();

			if (ImGui::BeginCombo("Tone Mapping", g_tone_mapping_operators[current_operator]))
			{
				for (int32_t i = 0; i < 5; i++)
				{
					if (ImGui::Selectable(g_tone_mapping_operators[i], current_operator == i))
						m_tone_mapping.set_current_operator(i);
				}
				ImGui::EndCombo();
			}
		}

		if (ImGui::CollapsingHeader("Settings"))
		{
			ImGui::SliderInt("Motion Blur Samples", &per_frame.max_motion_blur_samples, 1, 32);
			ImGui::SliderInt("SSAO Samples", &per_frame.ssao_num_samples, 1, 64);
			ImGui::SliderFloat("SSAO Radius", &per_frame.ssao_radius, 0.0f, 20.0f);
			ImGui::InputFloat("SSAO Bias", &per_frame.ssao_bias, 0.0f, 0.0f);

			float threshold = m_bloom.threshold();
			ImGui::SliderFloat("Bloom Threshold", &threshold, 0.0f, 2.0f);
			m_bloom.set_threshold(threshold);

			float strength = m_bloom.strength();
			ImGui::SliderFloat("Bloom Strength", &strength, 0.0f, 1.0f);
			m_bloom.set_strength(strength);
		}

		if (ImGui::CollapsingHeader("Profiling"))
		{
			m_shadow_map_renderer.profiling_gui();

			if (per_frame.renderer == RENDERER_FORWARD)
				m_forward_renderer.profiling_gui();
			else if (per_frame.renderer == RENDERER_DEFERRED)
			{
				m_gbuffer_renderer.profiling_gui();
				m_ambient_occlusion.profiling_gui();
				m_deferred_shading_renderer.profiling_gui();
			}

			m_ssr.profiling_gui();
			m_motion_blur.profiling_gui();
			m_bloom.profiling_gui();
			m_taa.profiling_gui();
		}

		ImGui::Separator();
		ImGui::Text("Press 'G' to toggle GUI");
	}
	ImGui::End();
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Renderer::render(double delta)
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
	update_uniforms(m_camera, delta);

	// Dispatch shadow map rendering.
	m_shadow_map_renderer.render(m_scene, &m_csm_technique);

	// Dispatch scene rendering.
	int renderer = GlobalGraphicsResources::per_frame_uniforms().renderer;

	if (renderer == RENDERER_FORWARD)
		m_forward_renderer.render(m_scene, m_width, m_height);
	else if (renderer == RENDERER_DEFERRED)
	{
		// Render geometry into G-Buffer
		m_gbuffer_renderer.render(m_scene, m_width, m_height);

		// Render SSAO
		m_ambient_occlusion.render(m_width, m_height);

		// Use G-Buffer and SSAO to perform deferred shading
		m_deferred_shading_renderer.render(m_scene, m_width, m_height);

		// Compute Screen Space Reflections
		m_ssr.render(m_width, m_height);
	}

	// TAA
	m_taa.render(m_width, m_height);

	// Motion blur
	m_motion_blur.render(m_width, m_height);

	// Bloom
	m_bloom.render(m_width, m_height);

	// Tone mapping
	m_tone_mapping.render(m_width, m_height);

	// Render final composition.
	m_final_composition.render(m_camera, m_width, m_height);
}

// -----------------------------------------------------------------------------------------------------------------------------------
