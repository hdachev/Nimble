#pragma once

#include <glm.hpp>
#include "macros.h"
#include "constants.h"

namespace nimble
{
	struct PointLightData
	{
		glm::vec4 position_range;
		glm::vec4 color_intensity;
		int32_t	  casts_shadow;
	};

	struct SpotLightData
	{
		glm::vec4 position_cone_angle;
		glm::vec4 direction_range;
		glm::vec4 color_intensity;
		int32_t	  casts_shadow;
	};

	struct DirectionalLightData
	{
		glm::vec4 direction;
		glm::vec4 color_intensity;
		int32_t	  casts_shadow;
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

	struct PerScenePointLightsUniforms
	{
		NIMBLE_ALIGNED(16) PointLightData 		point_lights[MAX_POINT_LIGHTS];
		NIMBLE_ALIGNED(16) int32_t			    point_light_count;
	};

	struct PerSceneSpotLightsUniforms
	{
		NIMBLE_ALIGNED(16) SpotLightData		spot_lights[MAX_SPOT_LIGHTS];
		NIMBLE_ALIGNED(16) int32_t			    spot_light_count;
	};

	struct PerSceneDirectionalLightsUniforms
	{
		NIMBLE_ALIGNED(16) DirectionalLightData directional_lights[MAX_DIRECTIONAL_LIGHTS];
		NIMBLE_ALIGNED(16) int32_t			    directional_light_count;
	};

	struct PerSceneUniforms
	{
		int32_t			     point_light_count;
		int32_t			     spot_light_count;
		int32_t			     directional_light_count;
		PointLightData 		 point_lights[MAX_POINT_LIGHTS];
		SpotLightData		 spot_lights[MAX_SPOT_LIGHTS];
		DirectionalLightData directional_lights[MAX_DIRECTIONAL_LIGHTS];
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