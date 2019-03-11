#pragma once

#include "../render_node.h"

namespace nimble
{
class HiZNode : public RenderNode
{
public:
	HiZNode(RenderGraph* graph);
	~HiZNode();
	
	void        declare_connections() override;
    bool        initialize(Renderer* renderer, ResourceManager* res_mgr) override;
    void        execute(double delta, Renderer* renderer, Scene* scene, View* view) override;
    void        shutdown() override;
    std::string name() override;
	void on_window_resized(const uint32_t& w, const uint32_t& h) override;

private:
	void copy_depth(Renderer* renderer, Scene* scene, View* view);
	void downsample(Renderer* renderer, Scene* scene, View* view);
	void create_rtvs();

private:
    std::shared_ptr<RenderTarget> m_depth_rt;
    std::shared_ptr<RenderTarget> m_hiz_rt;
    uint32_t m_num_rtv;
	RenderTargetView m_hiz_rtv[32];

	std::shared_ptr<Shader>  m_triangle_vs;

	std::shared_ptr<Shader>  m_hiz_fs;
	std::shared_ptr<Program> m_hiz_program;

	std::shared_ptr<Shader> m_copy_fs;
	std::shared_ptr<Program> m_copy_program;
};

DECLARE_RENDER_NODE_FACTORY(HiZNode);
} // namespace nimble