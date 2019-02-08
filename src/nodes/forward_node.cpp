#include "forward_node.h"
#include "../render_graph.h"

namespace nimble
{
	// -----------------------------------------------------------------------------------------------------------------------------------

	ForwardNode::ForwardNode(RenderGraph* graph) : SceneRenderNode(graph)
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	ForwardNode::~ForwardNode()
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void ForwardNode::execute_internal(const View* view)
	{
		//RenderTargetView views[] = { m_color_rtv, m_velocity_rtv };

		Params params;

		params.x = 0;
		params.y = 0;
		params.w = m_graph->window_width();
		params.h = m_graph->window_height();
		params.view = view;
		params.num_rt_views = 0;
		params.rt_views = nullptr;
		params.depth_views = nullptr;
		params.num_clear_colors = 1;

		render_scene(params);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	bool ForwardNode::register_resources()
	{
		//m_color_rt = register_render_target("Color", 1.0f, 1.0f, GL_TEXTURE_2D, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);
		//m_depth_rt = register_render_target("Depth", 1.0f, 1.0f, GL_TEXTURE_2D, GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT);
		//m_velocity_rt = register_render_target("Velocity", 1.0f, 1.0f, GL_TEXTURE_2D, GL_RG16F, GL_RG, GL_HALF_FLOAT);

		return true;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	bool ForwardNode::initialize()
	{		
		//m_color_rtv = RenderTargetView(0, 0, 0, m_color_rt.get());
		//m_velocity_rtv = RenderTargetView(0, 0, 0, m_velocity_rt.get());
		//m_depth_rtv = RenderTargetView(0, 0, 0, m_depth_rt.get());

		return true;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void ForwardNode::shutdown()
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	std::string ForwardNode::name()
	{
		return "Forward";
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	std::string ForwardNode::vs_template_path()
	{
		return "shader/forward/forward_vs.glsl";
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	std::string ForwardNode::fs_template_path()
	{
		return "shader/forward/forward_fs.glsl";
	}

	// -----------------------------------------------------------------------------------------------------------------------------------
}