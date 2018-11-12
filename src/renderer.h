#pragma once

#include <glm.hpp>
#include <unordered_map>
#include <memory>
#include "ogl.h"
#include "camera.h"
#include "macros.h"
#include "scene.h"
#include "forward_renderer.h"
#include "g_buffer_renderer.h"
#include "deferred_shading_renderer.h"
#include "ambient_occlusion.h"
#include "motion_blur.h"
#include "bloom.h"
#include "screen_space_reflections.h"
#include "tone_mapping.h"
#include "final_composition.h"
#include "uniforms.h"
#include "csm.h"
#include "shadow_map_renderer.h"
#include "taa.h"
#include "depth_of_field.h"
#include "hi_z_buffer.h"

namespace nimble
{
	class Renderer
	{
	public:
		Renderer();
		~Renderer();
		void initialize(uint16_t width, uint16_t height, Camera* camera);
		void shutdown();
		void set_scene(Scene* scene);
		Scene* scene();
		void debug_gui(double delta);
		void render(double delta);
		void set_camera(Camera* camera);
		void on_window_resized(uint16_t width, uint16_t height);

		inline PerSceneUniforms* per_scene_uniform() { return &m_per_scene_uniforms; }

	private:
		void update_uniforms(Camera* camera, double delta);

	private:
		Camera* m_camera;

		// Current window size.
		uint16_t m_width;
		uint16_t m_height;

		// Current scene.
		Scene* m_scene;

		// Renderers
		ForwardRenderer m_forward_renderer;
		GBufferRenderer m_gbuffer_renderer;
		DeferredShadingRenderer m_deferred_shading_renderer;
		FinalComposition m_final_composition;
		ShadowMapRenderer m_shadow_map_renderer;
		
		// Effects
		AmbientOcclusion m_ambient_occlusion;
		MotionBlur m_motion_blur;
		ScreenSpaceReflections m_ssr;
		Bloom	   m_bloom;
		ToneMapping m_tone_mapping;
		TAA			m_taa;
		DepthOfField m_depth_of_field;
		HiZBuffer	 m_hi_z_buffer;

		// CSM
		CSM m_csm_technique;

		// Uniform data.
		PerSceneUniforms m_per_scene_uniforms;
		PerEntityUniforms m_per_entity_uniforms[1024];
		PerMaterialUniforms m_per_material_uniforms[1024];

		// Debug options.
		glm::vec3 m_light_direction;
	};
}