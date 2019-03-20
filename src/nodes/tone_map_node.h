#pragma once

#include "../render_node.h"

namespace nimble
{
class ToneMapNode : public RenderNode
{
public:
    ToneMapNode(RenderGraph* graph);
    ~ToneMapNode();

    void        declare_connections() override;
    bool        initialize(Renderer* renderer, ResourceManager* res_mgr) override;
    void        execute(double delta, Renderer* renderer, Scene* scene, View* view) override;
    void        shutdown() override;
	std::string name() override;

private:
	int32_t m_tone_map_operator = 4;
	int32_t m_auto_exposure_type = 1;
	float m_key_value = 0.001f;
	float m_exposure = 0.0f;
    std::shared_ptr<Shader>       m_vs;
    std::shared_ptr<Shader>       m_fs;
    std::shared_ptr<Program>      m_program;
    std::shared_ptr<RenderTarget> m_texture;
	std::shared_ptr<RenderTarget> m_avg_luma;
};

DECLARE_RENDER_NODE_FACTORY(ToneMapNode);
} // namespace nimble
