#pragma once

#include "csm.h"
#include <glm.hpp>
#include <stdint.h>

namespace nimble
{
	struct DirectionalLight
	{
		using ID = uint32_t;

		ID m_id;
		glm::vec3 m_direction;
		glm::vec3 m_color;
		float m_intensity;
		CSM m_csm;
		bool m_casts_shadow;
	};

	struct PointLight
	{
		using ID = uint32_t;

		ID m_id;
		glm::vec3 m_position;
		glm::vec3 m_color;
		float m_intensity;
		std::shared_ptr<RenderTarget> m_render_target;
		bool m_casts_shadow;
	};

	struct SpotLight
	{
		using ID = uint32_t;

		ID m_id;
		glm::vec3 m_direction;
		glm::vec3 m_position;
		glm::vec3 m_color;
		float m_intensity;
		std::shared_ptr<RenderTarget> m_render_target;
		bool m_casts_shadow;
	};
} // namespace nimble