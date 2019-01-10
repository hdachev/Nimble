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

namespace nimble
{
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
			ShadowMapQuality shadow_map_quality = SHADOW_MAP_QUALITY_HIGH;
			uint32_t		 cascade_count = 4;
			uint32_t		 multisampling = 1;
		};

		Renderer(Settings settings = Settings());
		~Renderer();

		bool initialize();
		void render();
		void shutdown();

		void set_settings(Settings settings);
		void set_scene(std::shared_ptr<Scene> scene);
		void set_scene_render_graph(RenderGraph* graph);
		void queue_view(View view);
		void push_directional_light_views(View& dependent_view);
		void push_spot_light_views();
		void push_point_light_views();
		void clear_all_views();
		void on_window_resized(const uint32_t& w, const uint32_t& h);

		// Inline getters
		inline std::shared_ptr<Scene> scene() { return m_scene; }
		inline Settings settings() { return m_settings; }
		inline RenderGraph* scene_render_graph() { return m_scene_render_graph; }
		inline RenderTarget* directional_light_shadow_maps() { return m_directional_light_shadow_maps.get(); }
		inline RenderTarget* spot_light_shadow_maps() { return m_spot_light_shadow_maps.get(); }
		inline RenderTarget* point_light_shadow_maps() { return m_point_light_shadow_maps.get(); }

	private:
		void update_uniforms();
		void cull_scene();
		void queue_default_views();
		void render_all_views();

	private:
		// Current scene.
		uint32_t m_num_active_views = 0;
		std::array<View, MAX_VIEWS> m_active_views;
		std::array<Frustum, MAX_VIEWS> m_active_frustums;
		std::shared_ptr<Scene> m_scene;
		RenderGraph* m_scene_render_graph = nullptr;
		RenderGraph* m_shadow_map_render_graph = nullptr;
		std::array<PerViewUniforms, MAX_VIEWS> m_per_view_uniforms;
		std::array<PerEntityUniforms, MAX_ENTITIES> m_per_entity_uniforms;
		PerSceneUniforms m_per_scene_uniforms;

		// Shadow Maps
		std::shared_ptr<RenderTarget> m_directional_light_shadow_maps;
		std::shared_ptr<RenderTarget> m_spot_light_shadow_maps;
		std::shared_ptr<RenderTarget> m_point_light_shadow_maps;
		std::vector<RenderTargetView> m_directionl_light_rt_views;
		std::vector<RenderTargetView> m_point_light_rt_views;
		std::vector<RenderTargetView> m_spot_light_rt_views;
		Settings m_settings;
	};
}