#include "standard_depth_node.h"
#include "../render_graph.h"

namespace nimble
{
	// -----------------------------------------------------------------------------------------------------------------------------------

	StandardDepthNode::StandardDepthNode(RenderGraph* graph) : SceneRenderNode(graph)
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	StandardDepthNode::~StandardDepthNode()
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void StandardDepthNode::execute_internal(const View& view)
	{
		Params params;

		params.x = 0;
		params.y = 0;

		if (view.dest_render_target_view->texture->target() == GL_TEXTURE_2D)
		{
			Texture2D* texture = (Texture2D*)view.dest_render_target_view->texture.get();

			params.w = texture->width();
			params.h = texture->height();
		}
		else
		{
			TextureCube* texture = (TextureCube*)view.dest_render_target_view->texture.get();

			params.w = texture->width();
			params.h = texture->height();
		}

		params.view = &view;
		params.num_rt_views = 0;
		params.rt_views = nullptr;
		params.depth_views = view.dest_render_target_view;
		params.num_clear_colors = 1;

		render_scene(params);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	bool StandardDepthNode::register_resources()
	{
		return true;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	bool StandardDepthNode::initialize()
	{
		return true;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void StandardDepthNode::shutdown()
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	std::string StandardDepthNode::name()
	{
		return "Standard Depth Node";
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	std::string StandardDepthNode::vs_template_path()
	{
		return "shader/shadows/shadow_map/standard_depth_vs.glsl";
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	std::string StandardDepthNode::fs_template_path()
	{
		return "shader/shadows/shadow_map/standard_depth_fs.glsl";
	}

	// -----------------------------------------------------------------------------------------------------------------------------------
}