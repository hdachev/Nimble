#pragma once

#include <glm.hpp>
#include <unordered_map>
#include <memory>
#include <array>
#include "ogl.h"
#include "macros.h"
#include "scene.h"
#include "view.h"

namespace nimble
{
#define MAX_VIEWS 64

	class Renderer
	{
	public:
		Renderer();
		~Renderer();

		void render();

		void set_scene(std::shared_ptr<Scene> scene);
		void queue_view(View view);
		void push_directional_light_views(View& dependent_view);
		void push_spot_light_views();
		void push_point_light_views();
		void clear_all_views();

	private:
		void cull_scene();
		void queue_default_views();
		void render_all_views();

	private:
		// Current scene.
		uint32_t m_num_active_views = 0;
		std::array<View, MAX_VIEWS> m_active_views;
		std::shared_ptr<Scene> m_scene;
	};
}