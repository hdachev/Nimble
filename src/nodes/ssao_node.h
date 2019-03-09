#pragma once

#include "../render_node.h"

namespace nimble
{
class SSAONode : public RenderNode
{
public:
    SSAONode(RenderGraph* graph);
    ~SSAONode();

    void        declare_connections() override;
    bool        initialize(Renderer* renderer, ResourceManager* res_mgr) override;
    void        execute(Renderer* renderer, Scene* scene, View* view) override;
    void        shutdown() override;
    std::string name() override;

private:
    void ssao(Renderer* renderer, View* view);
    void blur(Renderer* renderer);

private:
    std::unique_ptr<Texture2D>     m_noise_texture;
    std::unique_ptr<UniformBuffer> m_kernel_ubo;

    RenderTargetView m_ssao_intermediate_rtv;
    RenderTargetView m_ssao_rtv;

    std::shared_ptr<RenderTarget> m_ssao_intermediate_rt;
    std::shared_ptr<RenderTarget> m_ssao_rt;

	std::shared_ptr<RenderTarget> m_normals_rt;
    std::shared_ptr<RenderTarget> m_depth_rt;

    std::shared_ptr<Shader> m_triangle_vs;

    std::shared_ptr<Shader>  m_ssao_fs;
    std::shared_ptr<Program> m_ssao_program;

    std::shared_ptr<Shader>  m_ssao_blur_fs;
    std::shared_ptr<Program> m_ssao_blur_program;

    int32_t m_num_samples = 64;
    float   m_radius      = 10.0f;
    float   m_bias        = 0.025f;
};

DECLARE_RENDER_NODE_FACTORY(SSAONode);
} // namespace nimble
