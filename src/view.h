#pragma once

#include <glm.hpp>
#include <stdint.h>
#include <memory>
#include "uniforms.h"

namespace nimble
{
	struct RenderTargetView;
	class RenderGraph;
	class Scene;

	struct View
	{
		bool enabled;
		bool culling;
		uint32_t id;
		glm::vec3 direction;
		glm::vec3 position;
		glm::mat4 view_mat;
		glm::mat4 projection_mat;
		glm::mat4 vp_mat;
		glm::mat4 prev_vp_mat;
		glm::mat4 inv_view_mat;
		glm::mat4 inv_projection_mat;
		glm::mat4 inv_vp_mat;
		glm::vec4 jitter;
		ShadowFrustum shadow_frustums[MAX_SHADOW_MAP_CASCADES * MAX_SHADOW_CASTING_DIRECTIONAL_LIGHTS];
		Scene* scene;
		std::shared_ptr<RenderGraph> graph;
		RenderTargetView* dest_render_target_view;
	};
} // namespace nimble
