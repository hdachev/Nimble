#include <../../common/uniforms.glsl>
#include <../../common/helper.glsl>

// ------------------------------------------------------------------
// DEFINES ----------------------------------------------------------
// ------------------------------------------------------------------

#define SSR_NUM_THREADS 8

// ------------------------------------------------------------------
// INPUTS -----------------------------------------------------------
// ------------------------------------------------------------------

layout (local_size_x = SSR_NUM_THREADS, local_size_y = SSR_NUM_THREADS) in;

// ------------------------------------------------------------------
// CONSTANTS --------------------------------------------------------
// ------------------------------------------------------------------

const float RAY_STEP_SIZE = 0.1;
const int 	MAX_RAY_MARCH_SAMPLES = 32;
const int 	MAX_BINARY_SEARCH_SAMPLES = 5;

// ------------------------------------------------------------------
// UNIFORMS ---------------------------------------------------------
// ------------------------------------------------------------------

layout (binding = 0, rgba32f) uniform image2D i_SSR;

uniform sampler2D s_HiZDepth;
uniform sampler2D s_Metallic;
uniform sampler2D s_Normal;

uniform float u_HiZLevels;

// ------------------------------------------------------------------
// FUNCTIONS --------------------------------------------------------
// ------------------------------------------------------------------

vec3 binary_search(in vec3 pos, in vec3 prev_pos)
{
	vec3 min_sample = prev_pos;
	vec3 max_sample = pos;
	vec3 mid_sample;

	for (int i = 0; i < MAX_BINARY_SEARCH_SAMPLES; i++)
	{
		mid_sample = mix(min_sample, max_sample, 0.5);
		float z_val = textureLod(s_HiZDepth, mid_sample.xy, 0).x;

		if (mid_sample.z > z_val)
			max_sample = mid_sample;
		else
			min_sample = mid_sample;
	}

	return mid_sample;
}

// ------------------------------------------------------------------

vec3 ray_march(in vec3 dir, in vec3 pos)
{
	vec3 prev_ray_sample = pos;

	for (int ray_step_idx = 0; ray_step_idx < MAX_RAY_MARCH_SAMPLES; ray_step_idx++)
	{
		vec3 ray_sample = (ray_step_idx * RAY_STEP_SIZE) * dir + pos;
		float z_val = textureLod(s_HiZDepth, ray_sample.xy, 0).x;

		if (ray_sample.z > z_val)
			return binary_search(ray_sample, prev_ray_sample);
		
		prev_ray_sample = ray_sample;
	}

	return vec3(1.0, 0.0, 0.0);
}

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
	ivec2 size = textureSize(s_HiZDepth, 0);
	vec2 tex_coord = vec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y) / vec2(size);

	// Fetch depth from hardware depth buffer
	float depth = textureLod(s_HiZDepth, tex_coord, 0).x;

	// Reconstruct world space position
	vec3 world_pos = world_position_from_depth(tex_coord, depth);

	// Get world space normal from G-Buffer
	vec3 normal = textureLod(s_Normal, tex_coord, 0).xyz;

	vec4 screen_pos = vec4(tex_coord, depth, 1.0);

	// Compute view direction
	vec3 view_dir = normalize(world_pos - view_pos.xyz);

	// Compute reflection vector
	vec3 reflection_dir = reflect(view_dir, normal);

	float camera_facing_refl_attenuation = 1.0 - smoothstep(0.25, 0.5, dot(-view_dir, reflection_dir));

	if (camera_facing_refl_attenuation <= 0.0)
		imageStore(i_SSR, ivec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y), vec4(0.0, 0.0, 0.0, 1.0));
	else
	{
		// Compute second screen space point in order to calculate SS reflection vector
		vec4 point_along_refl_dir = vec4(10.0 * reflection_dir + world_pos, 1.0);
		vec4 screen_space_refl_pos = view_proj * point_along_refl_dir;
		screen_space_refl_pos /= screen_space_refl_pos.w;
		screen_space_refl_pos.xyz = screen_space_refl_pos.xyz * vec3(0.5) + vec3(0.5);

		// Compute screen space reflection direction
		vec3 screen_space_refl_dir = normalize(screen_space_refl_pos.xyz - screen_pos.xyz);

		vec3 ssr = ray_march(screen_space_refl_dir.xyz, screen_pos.xyz);

		vec2 uv_sampling_attenuation = smoothstep(vec2(0.05), vec2(0.1), ssr.xy) * (vec2(1.0) - smoothstep(vec2(0.95), vec2(1.0), ssr.xy));
		uv_sampling_attenuation.x *= uv_sampling_attenuation.y;

		float total_attenuation = camera_facing_refl_attenuation * uv_sampling_attenuation.x;

		imageStore(i_SSR, ivec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y), vec4(ssr.xy, total_attenuation, 1.0));
	}
}

// ------------------------------------------------------------------