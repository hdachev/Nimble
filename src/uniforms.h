#pragma once

#include <glm.hpp>
#include "macros.h"
#include "constants.h"

namespace nimble
{
	struct PointLightData
	{
		NIMBLE_ALIGNED(16) glm::vec4 position_range;
		NIMBLE_ALIGNED(16) glm::vec4 color_intensity;
		NIMBLE_ALIGNED(16) int32_t	  casts_shadow;
	};

	struct SpotLightData
	{
		NIMBLE_ALIGNED(16) glm::vec4 position_cone_angle;
		NIMBLE_ALIGNED(16) glm::vec4 direction_range;
		NIMBLE_ALIGNED(16) glm::vec4 color_intensity;
		NIMBLE_ALIGNED(16) int32_t	  casts_shadow;
	};

	struct DirectionalLightData
	{
		NIMBLE_ALIGNED(16) glm::vec4 direction;
		NIMBLE_ALIGNED(16) glm::vec4 color_intensity;
		NIMBLE_ALIGNED(16) int32_t	  casts_shadow;
	};

	struct ShadowFrustum
	{
		NIMBLE_ALIGNED(16) glm::mat4 shadow_matrix;
		NIMBLE_ALIGNED(16) float	 far_plane;
	};

	// Size: 1392 bytes
	struct PerViewUniforms
	{
		NIMBLE_ALIGNED(16) glm::mat4	 last_view_proj;
		NIMBLE_ALIGNED(16) glm::mat4	 view_proj;
		NIMBLE_ALIGNED(16) glm::mat4	 inv_view_proj;
		NIMBLE_ALIGNED(16) glm::mat4	 inv_proj;
		NIMBLE_ALIGNED(16) glm::mat4	 inv_view;
		NIMBLE_ALIGNED(16) glm::mat4	 proj_mat;
		NIMBLE_ALIGNED(16) glm::mat4	 view_mat;
		NIMBLE_ALIGNED(16) glm::vec4	 view_pos;
		NIMBLE_ALIGNED(16) glm::vec4	 view_dir;
		NIMBLE_ALIGNED(16) glm::vec4	 current_prev_jitter;
		NIMBLE_ALIGNED(16) int			 num_cascades;
		NIMBLE_ALIGNED(16) ShadowFrustum shadow_frustums[MAX_SHADOW_MAP_CASCADES];
		float		 				 	 tan_half_fov;
		float		 				 	 aspect_ratio;
		float		 				 	 near_plane;
		float		 				 	 far_plane;
		// Renderer settings.
		int			 					 viewport_width;
		int			 					 viewport_height;
		uint8_t	  						 padding[212];
	};

	struct PerEntityUniforms
	{
		NIMBLE_ALIGNED(16) glm::mat4 modal_mat;
		NIMBLE_ALIGNED(16) glm::mat4 last_model_mat;
		NIMBLE_ALIGNED(16) glm::vec4 world_pos;
		uint8_t	  		   padding[112];
	};

	struct PerSceneUniforms
	{
		PointLightData 		 point_lights[MAX_POINT_LIGHTS];
		SpotLightData		 spot_lights[MAX_SPOT_LIGHTS];
		DirectionalLightData directional_lights[MAX_DIRECTIONAL_LIGHTS];
		int32_t			     point_light_count;
		int32_t			     spot_light_count;
		int32_t			     directional_light_count;
	};

	struct PerMaterialUniforms
	{
		NIMBLE_ALIGNED(16) glm::vec4 albedo;
		NIMBLE_ALIGNED(16) glm::vec4 emissive;
		NIMBLE_ALIGNED(16) glm::vec4 metalness_roughness;
	};

	struct PerFrustumSplitUniforms
	{
		NIMBLE_ALIGNED(16) glm::mat4 crop_matrix;
	};
}