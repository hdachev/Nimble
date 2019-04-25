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

uniform vec4 u_RayCastSize;
uniform float u_Thickness;
uniform int u_NumSteps;

// ------------------------------------------------------------------
// FUNCTIONS --------------------------------------------------------
// ------------------------------------------------------------------

float ssr_clamp(vec2 start, vec2 end, vec2 delta)
{
    vec2 dir = abs(end - start);
    return length(vec2(min(dir.x, delta.x), min(dir.y,delta.y)));
}

// ------------------------------------------------------------------

vec4 ray_march(vec3 view_dir, int num_steps, vec3 view_pos, vec3 screen_pos, vec2 uv, float thickness)
{
    // Compute a point along the reflection direction and project into clip space.
    vec4 ray_proj = proj_mat * vec4(view_pos + view_dir, 1.0);

    // Map the new point into NDC range (divide by w) and compute a screen space direction vector from current frag -> ray_proj.
    vec3 ray_dir = normalize(ray_proj.xyz / ray_proj.w - screen_pos);

    // Scale by 0.5 to map it to texture space.
    ray_dir.xy *= 0.5;

    // Compute ray start position in texture space while using NDC depth.
    vec3 ray_start = vec3(uv, screen_pos.z);

    // Texel size.
    vec2 screen_delta_2 = u_RayCastSize.zw;

    // Clamp ray step size to the texel size of the current mip level.
    float d = ssr_clamp(ray_start.xy, ray_start.xy + ray_dir.xy, screen_delta_2);

    // Compute first sample.
    vec3 sample_pos = ray_start + ray_dir * d;

    // Set Hi-Z mip level to the base mip (0).
    int level = 0;

    // Hit mask set to 0 initially since there are no hits.
    float mask = 0;
	
    // Begin ray march...
    for (int i = 0; i < num_steps; i++)
    {
        // Compute texel offset for the current mip level.
        vec2 current_delta = screen_delta_2 * exp2(level + 1);

        // Clamp ray step size to the texel size of the current mip level.
        float dist = ssr_clamp(sample_pos.xy, sample_pos.xy + ray_dir.xy, current_delta);

        // March ray forward.
        vec3 current_ray_pos = sample_pos + ray_dir * dist;

        // Sample scene depth from depth buffer at the current ray coordinates. 
#ifdef SSR_DEPTH_ZERO_TO_ONE
        // float scene_depth_at_ray_pos = textureLod(s_HiZDepth, current_ray_pos.xy, level).r;
#else
        float scene_depth_at_ray_pos = textureLod(s_HiZDepth, current_ray_pos.xy, level).r * 2.0 - 1.0;
#endif
        // Get depth of current ray position.
        float current_ray_depth = current_ray_pos.z;

        // If ray does NOT intersect the depth buffer, move to the next mip-level
        if (current_ray_depth < scene_depth_at_ray_pos)
        {
            level = min(level + 1, MAX_HIZ_LEVEL);
            sample_pos = current_ray_pos;
        }
        else // Else, go back a mip-level in order to check if the ray still intersects the depth buffer. If not, march the ray forward.
            level--;

        // If Hi-Z mip level is less than 0, we have converged the ray or could not find any intersections.
        if (level < 0)
        {
#ifdef SSR_DEPTH_ZERO_TO_ONE
            float delta = (-linear_eye_depth(scene_depth_at_ray_pos)) - (-linear_eye_depth(sample_pos.z));
#else
            float delta = (-linear_eye_depth(scene_depth_at_ray_pos * 0.5 + 0.5)) - (-linear_eye_depth(sample_pos.z * 0.5 + 0.5));
#endif
            mask = float(delta <= thickness && i > 0);
            return vec4(sample_pos, mask);
        }
    }

    return vec4(sample_pos, mask);
}

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
    vec2 uv = (vec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y) + vec2(0.5)) * u_RayCastSize.zw;

    vec3 world_normal = texture(s_Normal, uv).rgb;
    vec3 view_normal = world_to_view_space_normal(world_normal);

    float depth = texture(s_HiZDepth, uv).r;

#ifdef SSR_DEPTH_ZERO_TO_ONE
    vec3 screen_pos = vec3(uv * 2.0 - 1.0, depth);
#else
    vec3 screen_pos = vec3(uv, depth) * 2.0 - vec3(1.0);
#endif
    
    vec3 world_pos = world_position_from_depth(uv, depth);
    vec3 view_pos = view_position_from_depth(uv, depth);
    vec3 view_dir = normalize(view_pos);
    vec3 dir = reflect(view_dir, view_normal);

    // Retrieve metalness and roughness values
	float metallic = texture(s_Metallic, uv).r;

	float camera_facing_refl_attenuation = 1.0 - smoothstep(0.25, 0.5, dot(-view_dir, dir));

	if (camera_facing_refl_attenuation <= 0.0 || metallic < 0.1)
		imageStore(i_SSR, ivec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y), vec4(0.0, 0.0, 0.0, 1.0));
	else
	{
        vec4 hit_coord = ray_march(dir, u_NumSteps, view_pos, screen_pos, uv, u_Thickness);

		vec2 tc = smoothstep(0.2, 0.6, abs(vec2(0.5, 0.5) - hit_coord.xy));
		float screen_edge_factor = clamp(1.0 - (tc.x + tc.y), 0.0, 1.0);

		float total_attenuation = camera_facing_refl_attenuation * screen_edge_factor;

		imageStore(i_SSR, ivec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y), vec4(hit_coord.xy, total_attenuation, 1.0));
    }
}

// ------------------------------------------------------------------