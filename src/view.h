#pragma once

#include <glm.hpp>
#include <stdint.h>

namespace nimble
{
	class RenderTargetView;
	class RenderGraph;
	class Scene;

	struct View
	{
		bool m_enabled;
		bool m_culling;
		uint32_t m_id;
		glm::vec3 m_direction;
		glm::vec3 m_position;
		glm::mat4 m_view_mat;
		glm::mat4 m_projection_mat;
		glm::mat4 m_vp_mat;
		glm::mat4 m_prev_vp_mat;
		glm::mat4 m_inv_view_mat;
		glm::mat4 m_inv_projection_mat;
		glm::mat4 m_inv_vp_mat;
		glm::vec4 m_jitter;
		uint32_t m_render_target_array_slice;
		uint32_t m_render_target_cubemap_slice;
		Scene* m_scene;
		RenderGraph* m_graph;
		RenderTargetView* m_dest_render_target_view;
	};
} // namespace nimble
