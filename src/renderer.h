#pragma once

#include <glm.hpp>
#include <unordered_map>
#include <memory>
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
		View* allocate_view();
		void queue_view(View* view);
		void push_directional_light_views(View* dependent_view);
		void push_spot_light_views();
		void push_point_light_views();
		void clear_all_views();

	private:
		void cull_scene();
		void queue_default_views();
		void render_all_views();

	private:
		// Current scene.
		uint32_t m_num_allocated_views = 0;
		uint32_t m_num_active_views = 0;
		View m_view_pool[64];
		View* m_active_views[64];
		std::shared_ptr<Scene> m_scene;
	};
}