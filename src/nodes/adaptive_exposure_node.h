#pragma once

#include "../render_node.h"

namespace nimble
{
class AdaptiveExposureNode : public RenderNode
{
public:
	AdaptiveExposureNode(RenderGraph* graph);
	~AdaptiveExposureNode();
	
	void        declare_connections() override;
    bool        initialize(Renderer* renderer, ResourceManager* res_mgr) override;
    void        execute(double delta, Renderer* renderer, Scene* scene, View* view) override;
    void        shutdown() override;
    std::string name() override;

private:
	void initial_luminance(Renderer* renderer, Scene* scene, View* view);
	void adapted_luminance(double delta, Renderer* renderer, Scene* scene, View* view);
	void copy_luminance(Renderer* renderer, Scene* scene, View* view);

private:
	std::shared_ptr<RenderTarget> m_color_rt;

    std::shared_ptr<RenderTarget> m_luminance_rt;
	std::shared_ptr<RenderTarget> m_adapted_luminance_rt[2];
	std::shared_ptr<RenderTarget> m_final_luminance_rt;

	RenderTargetView m_luminance_rtv;
	RenderTargetView m_adapted_luminance_rtv[2];
	RenderTargetView m_final_luminance_rtv;

	std::shared_ptr<Shader>  m_vs;

	std::shared_ptr<Shader>  m_lum_fs;
	std::shared_ptr<Program> m_lum_program;

	std::shared_ptr<Shader>  m_adapted_lum_fs;
	std::shared_ptr<Program> m_adapted_lum_program;

	std::shared_ptr<Shader>  m_copy_lum_fs;
	std::shared_ptr<Program> m_copy_lum_program;

	float m_tau = 1.1f;
	int32_t m_current_rt = 0;
};

DECLARE_RENDER_NODE_FACTORY(AdaptiveExposureNode);
} // namespace nimble