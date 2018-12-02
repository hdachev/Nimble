#pragma once

#include <glm.hpp>
#include <stdint.h>

namespace nimble
{
	enum ViewType : uint32_t
	{
		VIEW_TYPE_SINGLE,
		VIEW_TYPE_MULTI
	};

	struct View
	{
		using ID = uint32_t;

		ViewType m_type;
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
	};
} // namespace nimble
