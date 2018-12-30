#include "forward_render_node.h"
#include "../render_graph.h"
#include "../global_graphics_resources.h"

namespace nimble
{
	// -----------------------------------------------------------------------------------------------------------------------------------

	ForwardRenderNode::ForwardRenderNode(RenderGraph* graph) : SceneRenderNode(graph)
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	ForwardRenderNode::~ForwardRenderNode()
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void ForwardRenderNode::execute(const View& view)
	{
		RenderTargetView views[] = { m_color_rtv, m_velocity_rtv };

		Params params;

		params.view = &view;
		params.num_rt_views = 2;
		params.rt_views = views;
		params.depth_views = &m_depth_rtv;
		params.num_clear_colors = 1;

		render_scene(params);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	bool ForwardRenderNode::initialize()
	{
		m_color_rt = register_render_target("Color", 1.0f, 1.0f, GL_TEXTURE_2D, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);
		m_depth_rt = register_render_target("Depth", 1.0f, 1.0f, GL_TEXTURE_2D, GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT);
		m_velocity_rt = register_render_target("Velocity", 1.0f, 1.0f, GL_TEXTURE_2D, GL_RG16F, GL_RG, GL_HALF_FLOAT);

		m_color_rtv = RenderTargetView(0, 0, 0, m_color_rt.get());
		m_velocity_rtv = RenderTargetView(0, 0, 0, m_velocity_rt.get());
		m_depth_rtv = RenderTargetView(0, 0, 0, m_depth_rt.get());

		return true;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void ForwardRenderNode::shutdown()
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	std::string ForwardRenderNode::name()
	{
		return "Forward Render Node";
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	std::string ForwardRenderNode::vs_template_path()
	{
		return "shader/forward/forward_vs.glsl";
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	std::string ForwardRenderNode::fs_template_path()
	{
		return "shader/forward/forward_fs.glsl";
	}

	// -----------------------------------------------------------------------------------------------------------------------------------
}