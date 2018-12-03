#pragma once

#include <glm.hpp>
#include <stdint.h>

namespace nimble
{
	struct RenderTarget;

	struct View
	{
		using ID = uint32_t;

		ID m_id;
		bool m_enabled;
		bool m_culling;
		glm::vec3 m_direction;
		glm::vec3 m_position;
		glm::mat4 m_view_mat;
		glm::mat4 m_projection_mat;
		glm::mat4 m_inv_view_mat;
		glm::mat4 m_inv_projection_mat;
		glm::mat4 m_inv_vp_mat;
		glm::vec4 m_jitter;
		RenderTarget* m_render_target;
	};
} // namespace nimble
