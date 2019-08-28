#pragma once

#include "../render_node.h"

#define BLOOM_TEX_CHAIN_SIZE 5

namespace nimble
{
class BloomNode : public RenderNode
{
public:
    BloomNode(RenderGraph* graph);
    ~BloomNode();

    void        declare_connections() override;
    bool        initialize_private(Renderer* renderer, ResourceManager* res_mgr) override;
    void        execute(double delta, Renderer* renderer, Scene* scene, View* view) override;
    void        shutdown() override;
    std::string name() override;

private:
    void bright_pass(Renderer* renderer);
    void downsample(Renderer* renderer);
    void upsample(Renderer* renderer);
    void composite(Renderer* renderer);

private:
    float m_threshold;
    float m_strength;
    bool  m_enabled;

    std::shared_ptr<RenderTarget> m_composite_rt;
    std::shared_ptr<RenderTarget> m_bloom_rt[BLOOM_TEX_CHAIN_SIZE];

    RenderTargetView m_composite_rtv;
    RenderTargetView m_bloom_rtv[BLOOM_TEX_CHAIN_SIZE];

    std::shared_ptr<Shader> m_triangle_vs;

    // Brightpass shader
    std::shared_ptr<Shader>  m_bright_pass_fs;
    std::shared_ptr<Program> m_bright_pass_program;

    // Downsample shader
    std::shared_ptr<Shader>  m_bloom_downsample_fs;
    std::shared_ptr<Program> m_bloom_downsample_program;

    // Upsample shader
    std::shared_ptr<Shader>  m_bloom_upsample_fs;
    std::shared_ptr<Program> m_bloom_upsample_program;

    // Composite shader
    std::shared_ptr<Shader>  m_bloom_composite_fs;
    std::shared_ptr<Program> m_bloom_composite_program;

    std::shared_ptr<RenderTarget> m_color_rt;
};

DECLARE_RENDER_NODE_FACTORY(BloomNode);
} // namespace nimble
