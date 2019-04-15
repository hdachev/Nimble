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

const float kRayStep = 0.1;
const float kMinRayStep = 0.1;
const int kMaxSteps = 30;
const float kSearchDist = 5.0;
const int kNumBinarySearchSteps = 5;
const float kReflectionSpecularFalloffExponent = 3.0;

// ------------------------------------------------------------------
// UNIFORMS ---------------------------------------------------------
// ------------------------------------------------------------------

layout (binding = 0, r32f) uniform image2D i_SSR;

uniform sampler2D s_HiZDepth;
uniform sampler2D s_Metallic;
uniform sampler2D s_Normal;

uniform float u_HiZLevels;

// ------------------------------------------------------------------
// FUNCTIONS --------------------------------------------------------
// ------------------------------------------------------------------

vec3 binary_search(inout vec3 dir, inout vec3 hit_coord, inout float out_depth)
{
	float depth;
	vec4 projected_coord;

	for (int i = 0; i < kNumBinarySearchSteps; i++)
	{
		projected_coord = projMat * vec4(hit_coord, 1.0);
		projected_coord.xy /= projected_coord.w;
		projected_coord.xy = projected_coord.xy * 0.5 + 0.5;

		depth = get_view_space_depth(projected_coord.xy, texture(s_Depth, projected_coord.xy).r);

		out_depth = hit_coord.z - depth;

		dir *= 0.5;

		if (out_depth > 0.0)
			hit_coord += dir;
		else
			hit_coord -= dir;
	}

	projected_coord = projMat * vec4(hit_coord, 1.0);
	projected_coord.xy /= projected_coord.w;
	projected_coord.xy = projected_coord.xy * 0.5 + 0.5;

	return vec3(projected_coord.xy, depth);
}

// ------------------------------------------------------------------

vec3 ray_march(in vec3 dir, in vec3 pos)
{
	for (int ray_step_idx = 0; ray_step_idx < kMaxSteps; ray_step_idx++)
	{
		vec3 ray_sample = (ray_step_idx * kRayStep) * dir + pos;
		float z_val = texture(s_Depth, ray_sample.xy);

		if (ray_sample.z > z_val)
		{

		}
	}
}

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
	ivec2 size = textureSize(s_HiZDepth, 0);
	vec2 tex_coord = vec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y) / vec2(size);

	// Fetch depth from hardware depth buffer
	float depth = textureLod(s_HiZDepth, tex_coord, 0);

	// Reconstruct world space position
	vec3 world_pos = world_position_from_depth(tex_coord, depth);

	// Get world space normal from G-Buffer
	vec3 normal = textureLod(s_Normal, tex_coord, 0);

	vec4 screen_pos = vec4(tex_coord, depth, 1.0);

	// Compute view direction
	vec3 view_dir = normalize(world_pos - view_pos.xyz);

	// Compute reflection vector
	vec3 reflection_dir = reflect(view_dir, normal);

	// Compute second screen space point in order to calculate SS reflection vector
	vec4 point_along_refl_dir = vec4(10.0 * reflection_dir + world_pos, 1.0);
	vec4 screen_space_refl_pos = view_proj * point_along_refl_dir;
	screen_space_refl_pos /= screen_space_refl_pos.w;
	screen_space_refl_pos.xy = screen_space_refl_pos.xy * vec2(0.5) + vec2(0.5);

	// Compute screen space reflection direction
	vec3 screen_space_refl_dir = normalize(screen_space_refl_pos.xyz - world_pos);

	vec3 ssr = ray_march(screen_space_refl_dir.xyz, screen_pos.xyz);

	imageStore(i_SSR, ivec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y), vec4(ssr, 1.0));
}

// ------------------------------------------------------------------