#pragma once

#include <glm.hpp>
#include <stdint.h>

namespace nimble
{
	struct Light
	{
		bool enabled;
		glm::vec3 color;
		float intensity;
		bool casts_shadow;
	};

	struct DirectionalLight : public Light
	{
		using ID = uint32_t;

		ID id;
		glm::vec3 rotation;
	};

	struct PointLight : public Light
	{
		using ID = uint32_t;

		ID id;
		glm::vec3 position;
	};

	struct SpotLight : public Light
	{
		using ID = uint32_t;

		ID id;
		glm::vec3 rotation;
		glm::vec3 position;
	};
} // namespace nimble