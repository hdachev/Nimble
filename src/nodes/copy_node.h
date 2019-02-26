#pragma once

#include "../render_node.h"

namespace nimble
{
class CopyNode : public RenderNode
{
public:
    CopyNode(RenderGraph* graph);
    ~CopyNode();

    void        declare_connections() override;
    bool        initialize(Renderer* renderer, ResourceManager* res_mgr) override;
    void        execute(Renderer* renderer, Scene* scene, View* view) override;
    void        shutdown() override;
    std::string name() override;

private:
    std::shared_ptr<Shader>       m_vs;
    std::shared_ptr<Shader>       m_fs;
    std::shared_ptr<Program>      m_program;
	std::shared_ptr<RenderTarget> m_texture;
};

DECLARE_RENDER_NODE_FACTORY(CopyNode);
} // namespace nimble
