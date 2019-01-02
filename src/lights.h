#pragma once

#include "csm.h"
#include <glm.hpp>
#include <stdint.h>

namespace nimble
{
	struct Light
	{
		glm::vec3 m_color;
		float m_intensity;
		bool m_casts_shadow;
	};

	struct DirectionalLight : public Light
	{
		using ID = uint32_t;

		ID m_id;
		glm::vec3 m_rotation;
	};

	struct PointLight : public Light
	{
		using ID = uint32_t;

		ID m_id;
		glm::vec3 m_position;
	};

	struct SpotLight : public Light
	{
		using ID = uint32_t;

		ID m_id;
		glm::vec3 m_rotation;
		glm::vec3 m_position;
	};
} // namespace nimble