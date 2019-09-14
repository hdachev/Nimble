#include "chromatic_aberration_node.h"

namespace nimble
{
DEFINE_RENDER_NODE_FACTORY(ChromaticAberrationNode)

// -----------------------------------------------------------------------------------------------------------------------------------

ChromaticAberrationNode::ChromaticAberrationNode(RenderGraph* graph) :
    RenderNode(graph)
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

ChromaticAberrationNode::~ChromaticAberrationNode()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void ChromaticAberrationNode::declare_connections()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool ChromaticAberrationNode::initialize(Renderer* renderer, ResourceManager* res_mgr)
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void ChromaticAberrationNode::execute(double delta, Renderer* renderer, Scene* scene, View* view)
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void ChromaticAberrationNode::shutdown()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::string ChromaticAberrationNode::name()
{
    return "Chromatic Aberration";
}

// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble