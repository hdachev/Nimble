#include <structures.glsl>

// ------------------------------------------------------------------
// COMMON UNIFORM BUFFERS -------------------------------------------
// ------------------------------------------------------------------

layout (std140) uniform u_PerFrame
{ 
	mat4 		  lastViewProj;
	mat4 		  viewProj;
	mat4 		  invViewProj;
	mat4 		  projMat;
	mat4 		  viewMat;
	vec4 		  viewPos;
	vec4 		  viewDir;
	int			  numCascades;
	ShadowFrustum shadowFrustums[MAX_SHADOW_FRUSTUM];
	float		  tanHalfFov;
	float		  aspectRatio;
	float		  nearPlane;
	float		  farPlane;
};

// ------------------------------------------------------------------

layout (std140) uniform u_PerScene
{
	PointLight 		 pointLights[MAX_POINT_LIGHTS];
	DirectionalLight directionalLight;
	int				 pointLightCount;
};

// ------------------------------------------------------------------

layout (std140) uniform u_PerEntity
{
	mat4 mvpMat;
	mat4 lastMvpMat;
	mat4 modelMat;	
	vec4 worldPos;
};

// ------------------------------------------------------------------