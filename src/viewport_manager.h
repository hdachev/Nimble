#pragma once

#include <vector>
#include <memory>

#include "viewport.h"

namespace nimble
{
	struct View;
	class Renderer;

	class ViewportManager
	{
    public:
		ViewportManager();
        ~ViewportManager();
        void initialize(const uint32_t& w, const uint32_t& h);
		void render_viewports(Renderer* renderer, uint32_t num_viewport, View** view);
        std::shared_ptr<Viewport> create_viewport(const std::string& name, float x_scale, float y_scale, float w_scale, float h_scale, int32_t z_order);
        std::shared_ptr<Viewport> lookup_viewport(const std::string& name);
        void                      on_window_resized(const uint32_t& w, const uint32_t& h);

	private:
        std::vector<std::shared_ptr<Viewport>> m_viewports;
        std::vector<View*>					   m_onscreen_views;
		uint32_t                               m_window_width;
		uint32_t                               m_window_height;
	};
}