#pragma once

#include "../global_probe_renderer.h"
#include "../render_target.h"
#include "bruneton_sky_model/bruneton_sky_model.h"

namespace nimble
{
class BrunetonProbeRenderer : public GlobalProbeRenderer
{
public:
    BrunetonProbeRenderer();
    ~BrunetonProbeRenderer();

    bool initialize(Renderer* renderer, ResourceManager* res_mgr) override;

protected:
    void        env_map(double delta, Renderer* renderer, Scene* scene) override;
    void        diffuse(double delta, Renderer* renderer, Scene* scene) override;
    void        specular(double delta, Renderer* renderer, Scene* scene) override;
    std::string probe_contribution_shader_path() override;

private:
    BrunetonSkyModel         m_sky_model;
    glm::mat4                m_cubemap_views[6];
    RenderTargetView         m_cubemap_rtv[6];
    std::string              m_scene_name = "";
    glm::vec3                m_sun_dir;
    std::shared_ptr<Shader>  m_env_map_vs;
    std::shared_ptr<Shader>  m_env_map_fs;
    std::shared_ptr<Program> m_env_map_program;
};
} // namespace nimble