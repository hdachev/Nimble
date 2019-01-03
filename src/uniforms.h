#pragma once

#include <glm.hpp>
#include "macros.h"

namespace nimble
{
	#define MAX_POINT_LIGHTS 32
	#define MAX_SHADOW_FRUSTUM 8

	struct PointLightUBO
	{
		NIMBLE_ALIGNED(16) glm::vec4 position;
		NIMBLE_ALIGNED(16) glm::vec4 color;
	};

	struct DirectionalLightUBO
	{
		NIMBLE_ALIGNED(16) glm::vec4 direction;
		NIMBLE_ALIGNED(16) glm::vec4 color;
	};

	struct ShadowFrustum
	{
		NIMBLE_ALIGNED(16) glm::mat4 shadowMatrix;
		NIMBLE_ALIGNED(16) float	 farPlane;
	};

	struct PerViewUniforms
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
		NIMBLE_ALIGNED(16) glm::vec4	 currentPrevJitter;
		NIMBLE_ALIGNED(16) int			 numCascades;
		NIMBLE_ALIGNED(16) ShadowFrustum shadowFrustums[MAX_SHADOW_FRUSTUM];
		float		 				 	 tanHalfFov;
		float		 				 	 aspectRatio;
		float		 				 	 nearPlane;
		float		 				 	 farPlane;
		// Renderer settings.
		int			 					 viewport_width;
		int			 					 viewport_height;
		uint8_t	  						 padding[212];
	};

	struct PerEntityUniforms
	{
		NIMBLE_ALIGNED(16) glm::mat4 modalMat;
		NIMBLE_ALIGNED(16) glm::mat4 lastModelMat;
		NIMBLE_ALIGNED(16) glm::vec4 worldPos;
		uint8_t	  		   padding[112];
	};

	struct PerSceneUniforms
	{
		NIMBLE_ALIGNED(16) PointLightUBO 		pointLights[MAX_POINT_LIGHTS];
		NIMBLE_ALIGNED(16) DirectionalLightUBO  directionalLight;
		NIMBLE_ALIGNED(16) int					pointLightCount;
	};

	struct PerMaterialUniforms
	{
		NIMBLE_ALIGNED(16) glm::vec4 albedo;
		NIMBLE_ALIGNED(16) glm::vec4 emissive;
		NIMBLE_ALIGNED(16) glm::vec4 metalnessRoughness;
	};

	struct PerFrustumSplitUniforms
	{
		NIMBLE_ALIGNED(16) glm::mat4 crop_matrix;
	};
}