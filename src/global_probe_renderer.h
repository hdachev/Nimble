#pragma once

#include "parameterizable.h"

namespace nimble
{
struct View;
class Scene;
class ResourceManager;
class Renderer;

class GlobalProbeRenderer : public Parameterizable
{
public:
    GlobalProbeRenderer();
    ~GlobalProbeRenderer();
    void render(double delta, Renderer* renderer, Scene* scene);

    virtual bool initialize(Renderer* renderer, ResourceManager* res_mgr);
    virtual void shutdown();

protected:
    virtual void        env_map(double delta, Renderer* renderer, Scene* scene)  = 0;
    virtual void        diffuse(double delta, Renderer* renderer, Scene* scene)  = 0;
    virtual void        specular(double delta, Renderer* renderer, Scene* scene) = 0;
    virtual std::string probe_contribution_shader_path()                         = 0;
};
} // namespace nimble