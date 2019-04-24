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

#ifdef OLD_SSR
const float RAY_STEP_SIZE = 0.1;
const int 	MAX_RAY_MARCH_SAMPLES = 32;
const int 	MAX_BINARY_SEARCH_SAMPLES = 16;
#endif

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

#ifdef OLD_SSR
float read_depth(vec2 uv)
{
    return textureLod(s_HiZDepth, uv.xy, 0).x;
}

vec3 binary_search(in vec3 pos, in vec3 prev_pos)
{
	vec3 min_sample = prev_pos;
	vec3 max_sample = pos;
	vec3 mid_sample;

	for (int i = 0; i < MAX_BINARY_SEARCH_SAMPLES; i++)
	{
		mid_sample = mix(min_sample, max_sample, 0.5);
		float z_val = read_depth(mid_sample.xy);

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
		vec3 ray_sample = (float(ray_step_idx) * RAY_STEP_SIZE) * dir + pos;
		float z_val = read_depth(ray_sample.xy);

		if (ray_sample.z > z_val)
			return binary_search(ray_sample, prev_ray_sample);
		
		prev_ray_sample = ray_sample;
	}

	return vec3(0.0, 0.0, 0.0);
}

vec3 trace_ray(vec3 ray_dir, vec3 ray_start)
{

    if (ray_dir.z < 0.0)
        return vec3(0.0, 0.0, 0.0);

    ray_dir = normalize(ray_dir);
    ivec2 work_size = textureSize(s_HiZDepth, 0);

    const int loop_max = 150;
    int mipmap = 0;
    int max_iter = loop_max;

    vec3 pos = ray_start;

    // Move pos by a small bias
    pos += ray_dir * 0.008;

    float hit_bias = 0.0017;

    while (mipmap > -1 && max_iter-- > 0)
    {

        // Check if we are out of screen bounds, if so, return
        if (pos.x < 0.0 || pos.y < 0.0 || pos.x > 1.0 || pos.y > 1.0 || pos.z < 0.0 || pos.z > 1.0)
            return vec3(0,0,0);

        // Fetch the current minimum cell plane height
        float cell_z = textureLod(s_HiZDepth, pos.xy, mipmap).x;

        // Compute the fractional part of the coordinate (scaled by the working size)
        // so the values will be between 0.0 and 1.0
        vec2 fract_coord = mod(pos.xy * work_size, 1.0);

        // Modify fract coord based on which direction we are stepping in.
        // Fract coord now contains the percentage how far we moved already in
        // the current cell in each direction.  
        fract_coord.x = ray_dir.x > 0.0 ? fract_coord.x : 1.0 - fract_coord.x;
        fract_coord.y = ray_dir.y > 0.0 ? fract_coord.y : 1.0 - fract_coord.y;

        // Compute maximum k and minimum k for which the ray would still be
        // inside of the cell.
        vec2 max_k_v = (1.0 / abs(ray_dir.xy)) / work_size.xy;
        vec2 min_k_v = -max_k_v * fract_coord.xy;

        // Scale the maximum k by the percentage we already processed in the current cell,
        // since e.g. if we already moved 50%, we can only move another 50%.
        max_k_v *= 1.0 - fract_coord.xy;

        // The maximum k is the minimum of the both sub-k's since if one component-maximum
        // is reached, the ray is out of the cell
        float max_k = min(max_k_v.x, max_k_v.y);

        // Same applies to the min_k, but because min_k is negative we have to use max()
        float min_k = max(min_k_v.x, min_k_v.y);

        // Check if the ray intersects with the cell plane. We have the following
        // equation: 
        // pos.z + k * ray_dir.z = cell.z
        // So k is:
        float k = (cell_z - pos.z) / ray_dir.z;

        // Optional: Abort when ray didn't exactly intersect:
        // if (k < min_k && mipmap <= 0) {
        //     return vec3(0);
        // } 

        // Check if we intersected the cell
        if (k < max_k + hit_bias)
        {
            // Clamp k
            k = max(min_k, k);

            if (mipmap < 1) 
            {
                pos += k * ray_dir;
                return pos;
            }

            // If we hit anything at a higher mipmap, step up to a higher detailed
            // mipmap:
            mipmap -= 2;
            work_size *= 4;
        } 
        else 
        {
            // If we hit nothing, move to the next cell, with a small bias
            pos += max_k * ray_dir * 1.04;
        }

        mipmap += 1;
        work_size /= 2;
    }

    return vec3(0.0, 0.0, 0.0);
}

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
	ivec2 size = textureSize(s_HiZDepth, 0);
	vec2 tex_coord = vec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y) / vec2(size.x - 1, size.y - 1);

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

    // Retrieve metalness and roughness values
	float metallic = texture(s_Metallic, tex_coord).r;

	float camera_facing_refl_attenuation = 1.0 - smoothstep(0.25, 0.5, dot(-view_dir, reflection_dir));

	if (camera_facing_refl_attenuation <= 0.0 || metallic < 0.1)
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

		vec3 ssr = trace_ray(screen_space_refl_dir.xyz, screen_pos.xyz);

		vec2 tex_coord = smoothstep(0.2, 0.6, abs(vec2(0.5, 0.5) - ssr.xy));
		float screen_edge_factor = clamp(1.0 - (tex_coord.x + tex_coord.y), 0.0, 1.0);

		float total_attenuation = camera_facing_refl_attenuation * screen_edge_factor;

		imageStore(i_SSR, ivec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y), vec4(ssr.xy, total_attenuation, 1.0));
	}
}
#endif

// ------------------------------------------------------------------

// ------------------------------------------------------------------
// CONSTANTS --------------------------------------------------------
// ------------------------------------------------------------------

#ifndef OLD_SSR
const float kRayStep = 0.1;
const float kMinRayStep = 0.1;
const int kMaxSteps = 32;
const float kSearchDist = 5.0;
const int kNumBinarySearchSteps = 5;
const float kReflectionSpecularFalloffExponent = 3.0;

// ------------------------------------------------------------------
// FUNCTIONS --------------------------------------------------------
// ------------------------------------------------------------------

vec3 binary_search(inout vec3 dir, inout vec3 hit_coord, inout float out_depth)
{
	float depth;
	vec4 projected_coord;

	for (int i = 0; i < kNumBinarySearchSteps; i++)
	{
		projected_coord = proj_mat * vec4(hit_coord, 1.0);
		projected_coord.xy /= projected_coord.w;
		projected_coord.xy = projected_coord.xy * 0.5 + 0.5;

		depth = linear_eye_depth(texture(s_HiZDepth, projected_coord.xy).r);

		out_depth = hit_coord.z - depth;

		dir *= 0.5;

		if (out_depth > 0.0)
			hit_coord += dir;
		else
			hit_coord -= dir;
	}

	projected_coord = proj_mat * vec4(hit_coord, 1.0);
	projected_coord.xy /= projected_coord.w;
	projected_coord.xy = projected_coord.xy * 0.5 + 0.5;

	return vec3(projected_coord.xy, depth);
}

// ------------------------------------------------------------------

vec3 ray_march(in vec3 dir, inout vec3 hit_coord, out float out_depth)
{
	dir *= kRayStep;

	float depth;
	vec4 projected_coord;

	for (int i = 0; i < kMaxSteps; i++)
	{
		hit_coord += dir;

		projected_coord = proj_mat * vec4(hit_coord, 1.0);
		projected_coord.xy /= projected_coord.w;
		projected_coord.xy = projected_coord.xy * 0.5 + 0.5;

		depth = linear_eye_depth(texture(s_HiZDepth, projected_coord.xy).r);

		if (depth > 1000.0)
			continue;

		out_depth = hit_coord.z - depth;

		if ((dir.z - out_depth) < 1.2)
		{
			if (out_depth <= 0.0)
			{
				vec3 result;
				result = binary_search(dir, hit_coord, out_depth);
				
				return result;
			}
		}
	}

	return vec3(projected_coord.xy, depth);
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

		vec3 hit_coord = ray_march(ray, hit_pos, out_depth);

		vec2 tex_coord = smoothstep(0.2, 0.6, abs(vec2(0.5, 0.5) - hit_coord.xy));
		float screen_edge_factor = clamp(1.0 - (tex_coord.x + tex_coord.y), 0.0, 1.0);
		// float reflection_multiplier = screen_edge_factor * -reflection.z;

		vec2 uv_sampling_attenuation = smoothstep(vec2(0.05), vec2(0.1), hit_coord.xy) * (vec2(1.0) - smoothstep(vec2(0.95), vec2(1.0), hit_coord.xy));
		uv_sampling_attenuation.x *= uv_sampling_attenuation.y;

		float total_attenuation = camera_facing_refl_attenuation * screen_edge_factor;//uv_sampling_attenuation.x;

		imageStore(i_SSR, ivec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y), vec4(hit_coord.xy, total_attenuation, 1.0));
	}
}
#endif

// ------------------------------------------------------------------