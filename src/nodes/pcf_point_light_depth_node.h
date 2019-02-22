#pragma once

#include "../render_node.h"

namespace nimble
{
class PCFPointLightDepthNode : public RenderNode
{
public:
    PCFPointLightDepthNode(RenderGraph* graph);
    ~PCFPointLightDepthNode();

    void		declare_connections() override;
    bool        initialize(Renderer* renderer, ResourceManager* res_mgr) override;
	void		execute(Renderer* renderer, Scene* scene, View* view) override;
    void        shutdown() override;
    std::string name() override;
    void        set_shader_uniforms(View* view, Program* program, int32_t& tex_unit);

private:
	std::shared_ptr<ShaderLibrary> m_library;
};

DECLARE_RENDER_NODE_FACTORY(PCFPointLightDepthNode);
} // namespace nimble