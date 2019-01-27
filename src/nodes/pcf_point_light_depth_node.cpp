#include "pcf_point_light_depth_node.h"
#include "../render_graph.h"

namespace nimble
{
	// -----------------------------------------------------------------------------------------------------------------------------------

	PCFPointLightDepthNode::PCFPointLightDepthNode(RenderGraph* graph) : SceneRenderNode(graph)
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	PCFPointLightDepthNode::~PCFPointLightDepthNode()
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void PCFPointLightDepthNode::execute_internal(const View& view)
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

	bool PCFPointLightDepthNode::register_resources()
	{
		return true;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	bool PCFPointLightDepthNode::initialize()
	{
		return true;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void PCFPointLightDepthNode::shutdown()
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	std::string PCFPointLightDepthNode::name()
	{
		return "PCF Point Light Depth Node";
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	std::string PCFPointLightDepthNode::vs_template_path()
	{
		return "shader/shadows/point_light/shadow_map/point_light_depth_vs.glsl";
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	std::string PCFPointLightDepthNode::fs_template_path()
	{
		return "shader/shadows/point_light/shadow_map/point_light_depth_fs.glsl";
	}

	// -----------------------------------------------------------------------------------------------------------------------------------
}