#include <structures.glsl>

// ------------------------------------------------------------------
// COMMON UNIFORM BUFFERS -------------------------------------------
// ------------------------------------------------------------------

layout (std140) uniform u_PerView
{ 
	mat4	 	  last_view_proj;
	mat4	 	  view_proj;
	mat4	 	  inv_view_proj;
	mat4	 	  inv_proj;
	mat4	 	  inv_view;
	mat4	 	  proj_mat;
	mat4	 	  view_mat;
	vec4	 	  view_pos;
	vec4	 	  view_dir;
	vec4	 	  current_prev_jitter;
	int			  num_cascades;
	ShadowFrustum shadow_frustums[MAX_SHADOW_MAP_CASCADES];
	float		  tan_half_fov;
	float		  aspect_ratio;
	float		  near_plane;
	float		  far_plane;
	int			  viewport_width;
	int			  viewport_height;
};

// ------------------------------------------------------------------

#ifdef POINT_LIGHTS
layout (std140) uniform u_PerScenePointLights
{
	PointLightData point_lights[MAX_POINT_LIGHTS];
	int			   point_light_count;
};
#endif

// ------------------------------------------------------------------

#ifdef SPOT_LIGHTS
layout (std140) uniform u_PerSceneSpotLights
{
	SpotLightData spot_lights[MAX_SPOT_LIGHTS];
	int		      spot_light_count;
};
#endif

// ------------------------------------------------------------------

#ifdef DIRECTIONAL_LIGHTS
layout (std140) uniform u_PerSceneDirectionalLights
{
	DirectionalLightData directional_lights[MAX_DIRECTIONAL_LIGHTS];
	int		    		 directional_light_count;
};
#endif

// ------------------------------------------------------------------

layout (std140) uniform u_PerEntity
{
	mat4 model_mat;
	mat4 last_model_mat;
	vec4 world_pos;
};

// ------------------------------------------------------------------

layout (std140) uniform u_PerSkeleton
{
	mat4 bone_transforms[MAX_BONES];
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