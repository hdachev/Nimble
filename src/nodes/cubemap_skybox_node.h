#pragma once

#include "../render_node.h"

namespace nimble
{
class CubemapSkyboxNode : public FullscreenRenderNode
{
public:
    CubemapSkyboxNode(RenderGraph* graph);
    ~CubemapSkyboxNode();

    bool        register_resources() override;
    bool        initialize() override;
    void        shutdown() override;
    std::string name() override;

private:
	void render_skybox(const View* view);

private:
    RenderTargetView m_scene_rtv;
    RenderTargetView m_depth_rtv;
	std::shared_ptr<Shader>			 m_vs;
	std::shared_ptr<Shader>		 m_fs;
	std::shared_ptr<Program>		 m_program;
};

DECLARE_RENDER_NODE_FACTORY(CubemapSkyboxNode);
} // namespace nimble#pragma once
