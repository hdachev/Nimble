#pragma once

#include <glm.hpp>
#include <macros.h>

#define MAX_POINT_LIGHTS 32
#define MAX_SHADOW_FRUSTUM 8

struct PointLight
{
	DW_ALIGNED(16) glm::vec4 position;
	DW_ALIGNED(16) glm::vec4 color;
};

struct DirectionalLight
{
	DW_ALIGNED(16) glm::vec4 direction;
	DW_ALIGNED(16) glm::vec4 color;
};

struct ShadowFrustum
{
	DW_ALIGNED(16) glm::mat4 shadowMatrix;
	DW_ALIGNED(16) float	 farPlane;
};

struct PerFrameUniforms
{
	DW_ALIGNED(16) glm::mat4	 lastViewProj;
	DW_ALIGNED(16) glm::mat4	 viewProj;
	DW_ALIGNED(16) glm::mat4	 invViewProj;
	DW_ALIGNED(16) glm::mat4	 projMat;
	DW_ALIGNED(16) glm::mat4	 viewMat;
	DW_ALIGNED(16) glm::vec4	 viewPos;
	DW_ALIGNED(16) glm::vec4	 viewDir;
	DW_ALIGNED(16) int			 numCascades;
	DW_ALIGNED(16) ShadowFrustum shadowFrustums[MAX_SHADOW_FRUSTUM];
	float		 tanHalfFov;
	float		 aspectRatio;
	float		 nearPlane;
	float		 farPlane;
	// Renderer settings.
	int			 renderer;
	int			 current_output;
	int			 motion_blur;
	int			 motion_blur_samples;
};

struct PerEntityUniforms
{
	DW_ALIGNED(16) glm::mat4 mvpMat;
	DW_ALIGNED(16) glm::mat4 lastMvpMat;
	DW_ALIGNED(16) glm::mat4 modalMat;
	DW_ALIGNED(16) glm::vec4 worldPos;
	uint8_t	  padding[48];
};

struct PerSceneUniforms
{
	DW_ALIGNED(16) PointLight 		pointLights[MAX_POINT_LIGHTS];
	DW_ALIGNED(16) DirectionalLight directionalLight;
	DW_ALIGNED(16) int				pointLightCount;
};

struct PerMaterialUniforms
{
	DW_ALIGNED(16) glm::vec4 albedoValue;
	DW_ALIGNED(16) glm::vec4 metalnessRoughness;
};

struct PerFrustumSplitUniforms
{
	DW_ALIGNED(16) glm::mat4 crop_matrix;
};
