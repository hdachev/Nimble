#include "viewport_manager.h"
#include "view.h"
#include "ogl.h"
#include "renderer.h"
#include "render_graph.h"

#include <algorithm>

namespace nimble
{
// -----------------------------------------------------------------------------------------------------------------------------------

bool viewport_sort_func(View* i, View* j)
{
    return (i->viewport->z_order < j->viewport->z_order);
}

// -----------------------------------------------------------------------------------------------------------------------------------

ViewportManager::ViewportManager()
{
    m_window_width  = 0;
    m_window_height = 0;
    m_viewports.reserve(32);
    m_onscreen_views.resize(32);
}

// -----------------------------------------------------------------------------------------------------------------------------------

ViewportManager::~ViewportManager()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void ViewportManager::initialize(const uint32_t& w, const uint32_t& h)
{
    m_window_width  = w;
    m_window_height = h;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void ViewportManager::render_viewports(Renderer* renderer, uint32_t num_viewport, View** views)
{
    uint32_t num_onscreen_views = 0;

    for (uint32_t i = 0; i < num_viewport; i++)
    {
        if (views[i]->viewport)
            m_onscreen_views[num_onscreen_views++] = views[i];
    }

    // Sort viewports by Z-Order
    std::sort(m_onscreen_views.begin(), m_onscreen_views.begin() + num_onscreen_views, viewport_sort_func);

    std::shared_ptr<Program> program = renderer->copy_program();

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    program->use();

    for (uint32_t i = 0; i < num_onscreen_views; i++)
    {
        const auto& view = m_onscreen_views[i];

        glViewport(view->viewport->x_scale * m_window_width, view->viewport->y_scale * m_window_height, view->viewport->w_scale * m_window_width, view->viewport->h_scale * m_window_height);

        if (program->set_uniform("s_Texture", 0))
        {
            auto& rt = view->graph->output_render_target();

            if (rt)
                rt->texture->bind(0);
        }

        // Render fullscreen triangle
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::shared_ptr<Viewport> ViewportManager::create_viewport(const std::string& name, float x_scale, float y_scale, float w_scale, float h_scale, int32_t z_order)
{
    for (auto& viewport : m_viewports)
    {
        if (viewport->name == name)
            return viewport;
    }

    std::shared_ptr<Viewport> viewport = std::make_shared<Viewport>();

    viewport->name    = name;
    viewport->x_scale = x_scale;
    viewport->y_scale = y_scale;
    viewport->w_scale = w_scale;
    viewport->h_scale = h_scale;
    viewport->z_order = z_order;

    m_viewports.push_back(viewport);

    return viewport;
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::shared_ptr<Viewport> ViewportManager::lookup_viewport(const std::string& name)
{
    for (auto& viewport : m_viewports)
    {
        if (viewport->name == name)
            return viewport;
    }

    return nullptr;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void ViewportManager::on_window_resized(const uint32_t& w, const uint32_t& h)
{
    m_window_width  = w;
    m_window_height = h;
}

// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble