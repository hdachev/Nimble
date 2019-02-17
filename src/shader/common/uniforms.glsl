#include <structures.glsl>

// ------------------------------------------------------------------
// COMMON UNIFORM BUFFERS -------------------------------------------
// ------------------------------------------------------------------

layout(std430, binding = 0) buffer u_PerView
{ 
	mat4	 	  last_view_proj;
	mat4	 	  view_proj;
	mat4	 	  inv_view_proj;
	mat4	 	  inv_proj;
	mat4	 	  inv_view;
	mat4	 	  proj_mat;
	mat4	 	  view_mat;
	mat4 		  cascade_matrix[MAX_SHADOW_CASTING_DIRECTIONAL_LIGHTS * MAX_SHADOW_MAP_CASCADES];
	vec4	 	  view_pos;
	vec4	 	  view_dir;
	vec4	 	  current_prev_jitter;
	float 		  cascade_far_plane[MAX_SHADOW_CASTING_DIRECTIONAL_LIGHTS * MAX_SHADOW_MAP_CASCADES];
	float		  tan_half_fov;
	float		  aspect_ratio;
	float		  near_plane;
	float		  far_plane;
	int			  num_cascades;
	int			  viewport_width;
	int			  viewport_height;
};

// ------------------------------------------------------------------

layout (std140) uniform u_PerEntity
{
	mat4 model_mat;
	mat4 last_model_mat;
};

// ------------------------------------------------------------------

layout(std430, binding = 2) buffer u_PerScene
{
	mat4 spot_light_shadow_matrix[MAX_SHADOW_CASTING_SPOT_LIGHTS];
    vec4 point_light_position_range[MAX_POINT_LIGHTS];
    vec4 point_light_color_intensity[MAX_POINT_LIGHTS];
    vec4 spot_light_position[MAX_SPOT_LIGHTS];
	vec4 spot_light_cutoff_inner_outer[MAX_SPOT_LIGHTS];
    vec4 spot_light_direction_range[MAX_SPOT_LIGHTS];
    vec4 spot_light_color_intensity[MAX_SPOT_LIGHTS];
    vec4 directional_light_direction[MAX_DIRECTIONAL_LIGHTS];
    vec4 directional_light_color_intensity[MAX_DIRECTIONAL_LIGHTS];
	int  point_light_casts_shadow[MAX_POINT_LIGHTS];
	int  spot_light_casts_shadow[MAX_SPOT_LIGHTS];
    int  directional_light_casts_shadow[MAX_DIRECTIONAL_LIGHTS];
    int  point_light_count;
    int  spot_light_count;
    int  directional_light_count;
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

#ifdef DIRECTIONAL_LIGHT_SHADOW_MAPPING
	uniform sampler2DArray   s_DirectionalLightShadowMaps;
#endif

#ifdef SPOT_LIGHT_SHADOW_MAPPING
	uniform sampler2DArray   s_SpotLightShadowMaps;
#endif

#ifdef POINT_LIGHT_SHADOW_MAPPING
	uniform samplerCubeArray s_PointLightShadowMaps;
#endif

// ------------------------------------------------------------------