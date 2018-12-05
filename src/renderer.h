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
	class Renderer
	{
	public:
		Renderer();
		~Renderer();

		void render();

		View* allocate_view();
		void push_view(View* view);
		void push_directional_light_views(View* dependent_view);
		void push_spot_light_views();
		void push_point_light_views();
		void clear_all_views();

	private:
		void cull_scene();
		void render_all_views();

	private:
		// Current scene.
		std::shared_ptr<Scene> m_scene;
	};
}