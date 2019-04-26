// https://github.com/simeonradivoev/ComputeStochasticReflections

#include <../../common/uniforms.glsl>
#include <../../common/helper.glsl>

// ------------------------------------------------------------------
// DEFINES ----------------------------------------------------------
// ------------------------------------------------------------------

#define SSR_NUM_THREADS 8
#define MAX_HIZ_LEVEL 6

// ------------------------------------------------------------------
// INPUTS -----------------------------------------------------------
// ------------------------------------------------------------------

layout (local_size_x = SSR_NUM_THREADS, local_size_y = SSR_NUM_THREADS) in;

// ------------------------------------------------------------------
// OUTPUTS ----------------------------------------------------------
// ------------------------------------------------------------------

layout (binding = 0, rgba32f) uniform image2D i_SSR;

// ------------------------------------------------------------------
// SAMPLERS ---------------------------------------------------------
// ------------------------------------------------------------------

uniform sampler2D s_HiZDepth;
uniform sampler2D s_Metallic;
uniform sampler2D s_Normal;

// ------------------------------------------------------------------
// UNIFORMS ---------------------------------------------------------
// ------------------------------------------------------------------

const float kRayStep = 0.1;
const float kMinRayStep = 0.1;
const int kMaxSteps = 32;
const float kSearchDist = 5.0;
const int kNumBinarySearchSteps = 5;
const float kReflectionSpecularFalloffExponent = 3.0;

// ------------------------------------------------------------------
// FUNCTIONS --------------------------------------------------------
// ------------------------------------------------------------------

float read_depth(vec2 uv)
{
    return textureLod(s_HiZDepth, uv.xy, 0).x * 2.0 - 1.0;
}

// ------------------------------------------------------------------

vec2 binary_search(inout vec3 dir, inout vec3 hit_coord)
{
	vec4 projected_coord;

	for (int i = 0; i < kNumBinarySearchSteps; i++)
	{
		projected_coord = proj_mat * vec4(hit_coord, 1.0);
		projected_coord.xyz /= projected_coord.w;
		projected_coord.xy = projected_coord.xy * 0.5 + 0.5;

		float depth = texture(s_HiZDepth, projected_coord.xy).r * 2.0 - 1.0;

		dir *= 0.5;

		if (projected_coord.z < depth)
			hit_coord += dir;
		else
			hit_coord -= dir;
	}

	projected_coord = proj_mat * vec4(hit_coord, 1.0);
	projected_coord.xy /= projected_coord.w;
	projected_coord.xy = projected_coord.xy * 0.5 + 0.5;

	return vec2(projected_coord.xy);
}

// ------------------------------------------------------------------

vec3 ray_march(in vec3 dir, in vec3 pos)
{
    vec3 ray_step = dir * kRayStep;
    vec3 ray_sample = pos;

	for (int ray_step_idx = 0; ray_step_idx < kMaxSteps; ray_step_idx++)
	{
        ray_sample += ray_step;

        vec4 projected_coord = proj_mat * vec4(ray_sample, 1.0);
		projected_coord.xyz /= projected_coord.w;
		projected_coord.xy = projected_coord.xy * 0.5 + 0.5;

		float z_val = read_depth(projected_coord.xy);

		if (projected_coord.z > z_val)
			return vec3(binary_search(ray_step, ray_sample), 1.0);
	}

	return vec3(0.0, 0.0, 0.0);
}

// ------------------------------------------------------------------

vec3 fresnel_schlick(float cos_theta, vec3 F0)
{	
	return F0 + (1.0 - F0) * pow(1.0 - cos_theta, 5.0);
}

// ------------------------------------------------------------------

#define Scale vec3(.8, .8, .8)
#define K 19.19

vec3 hash(vec3 a)
{
    a = fract(a * Scale);
    a += dot(a, a.yxz + K);
    return fract((a.xxy + a.yxx)*a.zyx);
}

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
	ivec2 size = textureSize(s_HiZDepth, 0);
	vec2 tex_coord = vec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y) / vec2(size.x - 1, size.y - 1);

	// Fetch depth from hardware depth buffer
	float depth = texture(s_HiZDepth, tex_coord).r;

	// Reconstruct view space position
	vec3 view_pos = view_position_from_depth(tex_coord, depth);

	// Get normal from G-Buffer in View Space
    vec3 world_normal = texture(s_Normal, tex_coord).rgb;
	vec3 view_normal = world_to_view_space_normal(world_normal);

	// Retrieve metalness and roughness values
	float metallic = texture(s_Metallic, tex_coord).r;

	vec3 view_dir = normalize(view_pos);

	// Calculate reflection vector
	vec3 reflection = normalize(reflect(view_dir, view_normal));

	float camera_facing_refl_attenuation = 1.0 - smoothstep(0.25, 0.5, dot(-view_dir, reflection));

	if (camera_facing_refl_attenuation <= 0.0 || metallic < 0.1)
		imageStore(i_SSR, ivec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y), vec4(0.0, 0.0, 0.0, 0.0));
	else
	{
		vec3 wp = vec3(inv_view * vec4(view_pos, 1.0));
		//vec3 jitt = mix(vec3(0.0), vec3(hash(wp)), roughness);

		vec3 hit_pos = view_pos;
		float out_depth;
		vec3 ray = reflection * max(kMinRayStep, -view_pos.z);

		vec3 hit_coord = ray_march(ray, hit_pos);

		vec2 tex_coord = smoothstep(0.2, 0.6, abs(vec2(0.5, 0.5) - hit_coord.xy));
		float screen_edge_factor = clamp(1.0 - (tex_coord.x + tex_coord.y), 0.0, 1.0);
		// float reflection_multiplier = screen_edge_factor * -reflection.z;

		vec2 uv_sampling_attenuation = smoothstep(vec2(0.05), vec2(0.1), hit_coord.xy) * (vec2(1.0) - smoothstep(vec2(0.95), vec2(1.0), hit_coord.xy));
		uv_sampling_attenuation.x *= uv_sampling_attenuation.y;

		float total_attenuation = camera_facing_refl_attenuation * screen_edge_factor * hit_coord.z;//uv_sampling_attenuation.x;

		imageStore(i_SSR, ivec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y), vec4(hit_coord.xy, total_attenuation, 1.0));
	}
}

// ------------------------------------------------------------------