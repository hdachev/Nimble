#include "pcf_directional_light_depth_node.h"
#include "../render_graph.h"

namespace nimble
{
	// -----------------------------------------------------------------------------------------------------------------------------------

	PCFDirectionalLightDepthNode::PCFDirectionalLightDepthNode(RenderGraph* graph) : SceneRenderNode(graph)
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	PCFDirectionalLightDepthNode::~PCFDirectionalLightDepthNode()
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void PCFDirectionalLightDepthNode::execute_internal(const View* view)
	{
		Params params;

		params.x = 0;
		params.y = 0;

		if (view->dest_render_target_view->texture->target() == GL_TEXTURE_2D)
		{
			Texture2D* texture = (Texture2D*)view->dest_render_target_view->texture.get();

			params.w = texture->height();
			params.h = texture->height();
		}
		else
		{
			TextureCube* texture = (TextureCube*)view->dest_render_target_view->texture.get();

			params.w = texture->height();
			params.h = texture->height();
		}

		params.view = view;
		params.num_rt_views = 0;
		params.rt_views = nullptr;
		params.depth_views = view->dest_render_target_view;
		params.num_clear_colors = 1;

		render_scene(params);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	bool PCFDirectionalLightDepthNode::register_resources()
	{
		return true;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	bool PCFDirectionalLightDepthNode::initialize()
	{
		return true;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void PCFDirectionalLightDepthNode::shutdown()
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	std::string PCFDirectionalLightDepthNode::name()
	{
		return "PCF Directional Light Depth";
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	std::string PCFDirectionalLightDepthNode::vs_template_path()
	{
		return "shader/shadows/directional_light/shadow_map/directional_light_depth_vs.glsl";
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	std::string PCFDirectionalLightDepthNode::fs_template_path()
	{
		return "shader/shadows/directional_light/shadow_map/directional_light_depth_fs.glsl";
	}

	// -----------------------------------------------------------------------------------------------------------------------------------
}