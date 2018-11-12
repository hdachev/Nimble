#pragma once

#include <glm.hpp>
#include "macros.h"

namespace nimble
{
	#define MAX_POINT_LIGHTS 32
	#define MAX_SHADOW_FRUSTUM 8

	struct PointLight
	{
		NIMBLE_ALIGNED(16) glm::vec4 position;
		NIMBLE_ALIGNED(16) glm::vec4 color;
	};

	struct DirectionalLight
	{
		NIMBLE_ALIGNED(16) glm::vec4 direction;
		NIMBLE_ALIGNED(16) glm::vec4 color;
	};

	struct ShadowFrustum
	{
		NIMBLE_ALIGNED(16) glm::mat4 shadowMatrix;
		NIMBLE_ALIGNED(16) float	 farPlane;
	};

	struct PerFrameUniforms
	{
		NIMBLE_ALIGNED(16) glm::mat4	 lastViewProj;
		NIMBLE_ALIGNED(16) glm::mat4	 viewProj;
		NIMBLE_ALIGNED(16) glm::mat4	 invViewProj;
		NIMBLE_ALIGNED(16) glm::mat4	 invProj;
		NIMBLE_ALIGNED(16) glm::mat4	 invView;
		NIMBLE_ALIGNED(16) glm::mat4	 projMat;
		NIMBLE_ALIGNED(16) glm::mat4	 viewMat;
		NIMBLE_ALIGNED(16) glm::vec4	 viewPos;
		NIMBLE_ALIGNED(16) glm::vec4	 viewDir;
		NIMBLE_ALIGNED(16) glm::vec4	 current_prev_jitter;
		NIMBLE_ALIGNED(16) int			 numCascades;
		NIMBLE_ALIGNED(16) ShadowFrustum shadowFrustums[MAX_SHADOW_FRUSTUM];
		float		 				 	 tanHalfFov;
		float		 				 	 aspectRatio;
		float		 				 	 nearPlane;
		float		 				 	 farPlane;
		// Renderer settings.
		int			 					 renderer;
		int			 					 current_output;
		int			 					 motion_blur;
		int			 					 max_motion_blur_samples;
		float		 					 velocity_scale;
		int			 					 viewport_width;
		int			 					 viewport_height;
		int			 					 ssao;
		int			 					 ssao_num_samples;
		float		 					 ssao_radius;
		float		 					 ssao_bias;
	};

	struct PerEntityUniforms
	{
		NIMBLE_ALIGNED(16) glm::mat4 mvpMat;
		NIMBLE_ALIGNED(16) glm::mat4 lastMvpMat;
		NIMBLE_ALIGNED(16) glm::mat4 modalMat;
		NIMBLE_ALIGNED(16) glm::vec4 worldPos;
		uint8_t	  		   padding[48];
	};

	struct PerSceneUniforms
	{
		NIMBLE_ALIGNED(16) PointLight 		pointLights[MAX_POINT_LIGHTS];
		NIMBLE_ALIGNED(16) DirectionalLight directionalLight;
		NIMBLE_ALIGNED(16) int				pointLightCount;
	};

	struct PerMaterialUniforms
	{
		NIMBLE_ALIGNED(16) glm::vec4 albedoValue;
		NIMBLE_ALIGNED(16) glm::vec4 metalnessRoughness;
	};

	struct PerFrustumSplitUniforms
	{
		NIMBLE_ALIGNED(16) glm::mat4 crop_matrix;
	};
}