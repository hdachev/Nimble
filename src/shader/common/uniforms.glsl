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
	vec4		  z_buffer_params;
	vec4		  time_params;
	vec4		  viewport_params;
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
	LightData  lights[MAX_LIGHTS];
    mat4  shadow_matrices[MAX_SHADOW_CASTING_SPOT_LIGHTS + MAX_SHADOW_CASTING_DIRECTIONAL_LIGHTS * MAX_SHADOW_MAP_CASCADES];
    uvec4 light_count;
};

// ------------------------------------------------------------------

uint total_light_count()
{
	return light_count.x;
}

// ------------------------------------------------------------------

uint directional_light_count()
{
	return light_count.y;
}

// ------------------------------------------------------------------

uint spot_light_count()
{
	return light_count.z;
}

// ------------------------------------------------------------------

uint point_light_count()
{
	return light_count.w;
}

// ------------------------------------------------------------------

uint spot_light_offset()
{
	return directional_light_count();
}

// ------------------------------------------------------------------

uint point_light_offset()
{
	return directional_light_count() + spot_light_count();
}

// ------------------------------------------------------------------

int light_type(uint light_idx)
{
	return lights[light_idx].indices0.x; 
}

// ------------------------------------------------------------------

int spot_light_shadow_map_index(uint light_idx)
{
	return lights[light_idx].indices0.y;
}

// ------------------------------------------------------------------

int spot_light_shadow_matrix_index(uint light_idx)
{
	return lights[light_idx].indices0.z;
}

// ------------------------------------------------------------------

vec3 spot_light_position(uint light_idx)
{
	return lights[light_idx].data0.xyz;
}

// ------------------------------------------------------------------

float spot_light_shadow_bias(uint light_idx)
{
	return lights[light_idx].data0.w;
}

// ------------------------------------------------------------------

float spot_light_inner_cutoff(uint light_idx)
{
	return lights[light_idx].data1.x;
}

// ------------------------------------------------------------------

float spot_light_outer_cutoff(uint light_idx)
{
	return lights[light_idx].data1.y;
}

// ------------------------------------------------------------------

vec3 spot_light_direction(uint light_idx)
{
	return lights[light_idx].data2.xyz;
}

// ------------------------------------------------------------------

float spot_light_range(uint light_idx)
{
	return lights[light_idx].data2.w;
}

// ------------------------------------------------------------------

vec3 spot_light_color(uint light_idx)
{
	return lights[light_idx].data3.xyz;
}

// ------------------------------------------------------------------

float spot_light_intensity(uint light_idx)
{
	return lights[light_idx].data3.w;
}

// ------------------------------------------------------------------

int directional_light_first_shadow_map_index(uint light_idx)
{
	return lights[light_idx].indices0.y;
}

// ------------------------------------------------------------------

int directional_light_first_shadow_matrix_index(uint light_idx)
{
	return lights[light_idx].indices0.z;
}

// ------------------------------------------------------------------

vec3 directional_light_direction(uint light_idx)
{
	return lights[light_idx].data0.xyz;
}

// ------------------------------------------------------------------

float directional_light_shadow_bias(uint light_idx)
{
	return lights[light_idx].data0.w;
}

// ------------------------------------------------------------------

vec3 directional_light_color(uint light_idx)
{
	return lights[light_idx].data1.xyz;
}

// ------------------------------------------------------------------

float directional_light_intensity(uint light_idx)
{
	return lights[light_idx].data1.w;
}

// ------------------------------------------------------------------

vec4 directional_light_cascade_far_planes(uint light_idx)
{
	return lights[light_idx].data2;
}

// ------------------------------------------------------------------

int point_light_shadow_map_index(uint light_idx)
{
	return lights[light_idx].indices0.y;
}

// ------------------------------------------------------------------

vec3 point_light_position(uint light_idx)
{
	return lights[light_idx].data0.xyz;
}

// ------------------------------------------------------------------

float point_light_shadow_bias(uint light_idx)
{
	return lights[light_idx].data0.w;
}

// ------------------------------------------------------------------

float point_light_near_field(uint light_idx)
{
	return lights[light_idx].data1.x;
}

// ------------------------------------------------------------------

float point_light_far_field(uint light_idx)
{
	return lights[light_idx].data1.y;
}

// ------------------------------------------------------------------

vec3 point_light_color(uint light_idx)
{
	return lights[light_idx].data2.xyz;
}

// ------------------------------------------------------------------

float point_light_intensity(uint light_idx)
{
	return lights[light_idx].data2.w;
}

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
	uniform vec4 u_Albedo;
#endif

#if defined(UNIFORM_METAL_SPEC) || defined(UNIFORM_ROUGH_SMOOTH)
	uniform vec4 u_MetalRough;
#endif

#ifdef UNIFORM_EMISSIVE
	uniform vec4 u_Emissive;
#endif

#ifdef DIRECTIONAL_LIGHT_SHADOW_MAPPING
	#ifdef USE_SHADOW_SAMPLERS_DIRECTIONAL
		uniform sampler2DArrayShadow s_DirectionalLightShadowMaps;
	#else
		uniform sampler2DArray s_DirectionalLightShadowMaps;
	#endif
#endif

#ifdef SPOT_LIGHT_SHADOW_MAPPING
	#ifdef USE_SHADOW_SAMPLERS_SPOT
		uniform sampler2DArrayShadow s_SpotLightShadowMaps;
	#else
		uniform sampler2DArray s_SpotLightShadowMaps;
	#endif
#endif

#ifdef POINT_LIGHT_SHADOW_MAPPING
	#ifdef USE_SHADOW_SAMPLERS_POINT
		uniform samplerCubeArrayShadow s_PointLightShadowMaps;
	#else
		uniform samplerCubeArray s_PointLightShadowMaps;
	#endif
#endif

// ------------------------------------------------------------------