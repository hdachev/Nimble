#include "bruneton_probe_renderer.h"
#include "../renderer.h"
#include "../resource_manager.h"
#include "../logger.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <gtc/matrix_transform.hpp>

namespace nimble
{
// -----------------------------------------------------------------------------------------------------------------------------------

BrunetonProbeRenderer::BrunetonProbeRenderer()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

BrunetonProbeRenderer::~BrunetonProbeRenderer()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool BrunetonProbeRenderer::initialize(Renderer* renderer, ResourceManager* res_mgr)
{
    bool status = m_sky_model.initialize(renderer, res_mgr);

    glm::mat4 capture_projection = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 10.0f);
    glm::mat4 capture_views[]    = {
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f)),
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, -1.0f, 0.0f))
    };

    for (int i = 0; i < 6; i++)
        m_cubemap_views[i] = capture_projection * capture_views[i];

    m_sun_dir = glm::vec3(0.0f);

    m_env_map_vs = res_mgr->load_shader("shader/sky_models/bruneton/env_map_vs.glsl", GL_VERTEX_SHADER);
    m_env_map_fs = res_mgr->load_shader("shader/sky_models/bruneton/env_map_fs.glsl", GL_FRAGMENT_SHADER);

    if (m_env_map_vs && m_env_map_fs)
    {
        m_env_map_program = renderer->create_program(m_env_map_vs, m_env_map_fs);

        if (!m_env_map_program)
        {
            NIMBLE_LOG_ERROR("Failed to create program");
            return false;
        }
    }
    else
    {
        NIMBLE_LOG_ERROR("Failed to load shaders");
        return false;
    }

    return status;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void BrunetonProbeRenderer::env_map(double delta, Renderer* renderer, Scene* scene)
{
    uint32_t          num_lights = scene->directional_light_count();
    DirectionalLight* lights     = scene->directional_lights();

    if (num_lights > 0)
    {
        DirectionalLight& light   = lights[0];
        glm::vec3         sun_dir = light.transform.forward();

        if (m_sun_dir != sun_dir || m_scene_name != scene->name())
        {
            m_scene_name = scene->name();
            m_sun_dir = sun_dir;

            for (int i = 0; i < 6; i++)
                m_cubemap_rtv[i] = RenderTargetView(i, 0, 0, scene->env_map());

            glDisable(GL_DEPTH_TEST);
            glDisable(GL_CULL_FACE);

            m_env_map_program->use();

            m_sky_model.set_render_uniforms(m_env_map_program.get(), -light.transform.forward());

            for (int i = 0; i < 6; i++)
            {
                m_env_map_program->set_uniform("view_projection", m_cubemap_views[i]);

                renderer->bind_render_targets(1, &m_cubemap_rtv[i], nullptr);
                glViewport(0, 0, ENVIRONMENT_MAP_SIZE, ENVIRONMENT_MAP_SIZE);

                glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
                glClear(GL_COLOR_BUFFER_BIT);

                renderer->cube_vao()->bind();

                glDrawArrays(GL_TRIANGLES, 0, 36);
            }
        }
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void BrunetonProbeRenderer::diffuse(double delta, Renderer* renderer, Scene* scene)
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void BrunetonProbeRenderer::specular(double delta, Renderer* renderer, Scene* scene)
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::string BrunetonProbeRenderer::probe_contribution_shader_path()
{
    return "";
}

// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble