#include <structures.glsl>

// ------------------------------------------------------------------
// COMMON UNIFORM BUFFERS -------------------------------------------
// ------------------------------------------------------------------

layout (std140) uniform u_PerView
{ 
	mat4	 	  lastViewProj;
	mat4	 	  viewProj;
	mat4	 	  invViewProj;
	mat4	 	  invProj;
	mat4	 	  invView;
	mat4	 	  projMat;
	mat4	 	  viewMat;
	vec4	 	  viewPos;
	vec4	 	  viewDir;
	vec4	 	  current_prev_jitter;
	int			  numCascades;
	ShadowFrustum shadowFrustums[MAX_SHADOW_FRUSTUM];
	float		  tanHalfFov;
	float		  aspectRatio;
	float		  nearPlane;
	float		  farPlane;
	int			  viewport_width;
	int			  viewport_height;
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
	mat4 modelMat;
	mat4 lastModelMat;
	vec4 worldPos;
};

// ------------------------------------------------------------------

layout (std140) uniform u_PerSkeleton
{
	mat4 boneTransforms[MAX_BONES];
};

// ------------------------------------------------------------------
// SAMPLERS  --------------------------------------------------------
// ------------------------------------------------------------------

#ifdef TEXTURE_ALBEDO
    uniform sampler2D s_Albedo;
#endif

#ifdef TEXTURE_NORMAL
    uniform sampler2D s_Normal;
#endif

#ifdef TEXTURE_METAL_SPEC
    uniform sampler2D s_MetalSpec;
#endif

#ifdef TEXTURE_ROUGH_SMOOTH
    uniform sampler2D s_RoughSmooth;
#endif

#ifdef TEXTURE_DISPLACEMENT
    uniform sampler2D s_Displacement;
#endif

#ifdef TEXTURE_EMISSIVE
    uniform sampler2D s_Emissive;
#endif

#ifdef UNIFORM_ALBEDO
	vec4 u_Albedo;
#endif

#ifdef UNIFORM_METAL_SPEC
	#ifdef UNIFORM_ROUGH_SMOOTH
		vec4 u_MetalRough;
	#else
		vec4 u_MetalRough;
	#endif
#else
	#ifdef UNIFORM_ROUGH_SMOOTH
		vec4 u_MetalRough;
	#endif
#endif

#ifdef UNIFORM_EMISSIVE
	vec4 u_Emissive;
#endif

// ------------------------------------------------------------------