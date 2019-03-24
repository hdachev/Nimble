#include "global_probe_renderer.h"

namespace nimble
{
// -----------------------------------------------------------------------------------------------------------------------------------

GlobalProbeRenderer::GlobalProbeRenderer()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

GlobalProbeRenderer::~GlobalProbeRenderer()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void GlobalProbeRenderer::render(double delta, Renderer* renderer, Scene* scene)
{
    env_map(delta, renderer, scene);
    diffuse(delta, renderer, scene);
    specular(delta, renderer, scene);
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool GlobalProbeRenderer::initialize(Renderer* renderer, ResourceManager* res_mgr)
{
    return true;
}

// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble