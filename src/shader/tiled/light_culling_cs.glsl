#include <../common/uniforms.glsl>
#include <../common/helper.glsl>

// ------------------------------------------------------------------
// DEFINES ----------------------------------------------------------
// ------------------------------------------------------------------

#define TILE_SIZE 16
#define MAX_POINT_LIGHTS_PER_TILE 512
#define MAX_SPOT_LIGHTS_PER_TILE 512

// ------------------------------------------------------------------
// STRUCTURES -------------------------------------------------------
// ------------------------------------------------------------------

struct LightIndices
{
    uint num_point_lights;
    uint num_spot_lights;
    uint point_light_indices[MAX_POINT_LIGHTS_PER_TILE];
    uint spot_light_indices[MAX_SPOT_LIGHTS_PER_TILE];
};

// ------------------------------------------------------------------

struct Frustum
{
    vec4 planes[6];
    vec4 points[8];
};

// ------------------------------------------------------------------
// INPUTS -----------------------------------------------------------
// ------------------------------------------------------------------

layout (local_size_x = TILE_SIZE, local_size_y = TILE_SIZE, local_size_z = 1) in;

// ------------------------------------------------------------------
// UNIFORMS ---------------------------------------------------------
// ------------------------------------------------------------------

layout(std430, binding = 3) buffer u_LightIndices
{
	LightIndices indices[];
};

uniform sampler2D s_Depth;

// ------------------------------------------------------------------
// SHARED DATA ------------------------------------------------------
// ------------------------------------------------------------------

shared uint g_MinDepth;
shared uint g_MaxDepth;
shared Frustum g_Frustum;
shared uint g_PointLightCount;
shared uint g_SpotLightCount;

// ------------------------------------------------------------------
// FUNCTIONS --------------------------------------------------------
// ------------------------------------------------------------------

vec4 screen_to_world_space(vec4 p, vec2 screen_size, mat4 inv_view_proj)
{
    vec2 tex_coord = p.xy / screen_size;
 
    vec4 world_pos = inv_view_proj * vec4(2.0 * tex_coord - 1.0, p.z, p.w); 
 
    return world_pos / world_pos.w;
}

vec4 compute_plane(vec3 p0, vec3 p1, vec3 p2)
{
    vec4 plane;
 
    vec3 v0 = p1 - p0;
    vec3 v2 = p2 - p0;
 
    plane.xyz = normalize(cross(v0, v2));
 
    // Compute the distance to the origin using p0.
    plane.w = dot(plane.xyz, p0);
 
    return plane;
}

Frustum create_frustum(uvec2 idx, uint min_depth, uint max_depth)
{
    vec4 screen_pos[8];

    float fmin_depth = 2.0 * uintBitsToFloat(min_depth) - 1.0;
    float fmax_depth = 2.0 * uintBitsToFloat(max_depth) - 1.0;
 
    screen_pos[0] = vec4(gl_WorkGroupID.xy * TILE_SIZE, fmax_depth, 1.0);                                // Far-Top-Left
    screen_pos[1] = vec4(vec2(gl_WorkGroupID.x + 1, gl_WorkGroupID.y) * TILE_SIZE, fmax_depth, 1.0);     // Far-Top-Right
    screen_pos[2] = vec4(vec2(gl_WorkGroupID.x, gl_WorkGroupID.y + 1)  * TILE_SIZE, fmax_depth, 1.0);    // Far-Bottom-Left
    screen_pos[3] = vec4(vec2(gl_WorkGroupID.x + 1, gl_WorkGroupID.y + 1) * TILE_SIZE, fmax_depth, 1.0); // Far-Bottom-Right

    screen_pos[4] = vec4(gl_WorkGroupID.xy * TILE_SIZE, fmin_depth, 1.0);                                // Near-Top-Left
    screen_pos[5] = vec4(vec2(gl_WorkGroupID.x + 1, gl_WorkGroupID.y) * TILE_SIZE, fmin_depth, 1.0);     // Near-Top-Right
    screen_pos[6] = vec4(vec2(gl_WorkGroupID.x, gl_WorkGroupID.y + 1)  * TILE_SIZE, fmin_depth, 1.0);    // Near-Bottom-Left
    screen_pos[7] = vec4(vec2(gl_WorkGroupID.x + 1, gl_WorkGroupID.y + 1) * TILE_SIZE, fmin_depth, 1.0); // Near-Bottom-Right
 
    vec4 world_pos[8];
 
    world_pos[0] = screen_to_world_space(screen_pos[0], viewport_params.xy, inv_view_proj);
    world_pos[1] = screen_to_world_space(screen_pos[1], viewport_params.xy, inv_view_proj);
    world_pos[2] = screen_to_world_space(screen_pos[2], viewport_params.xy, inv_view_proj);
    world_pos[3] = screen_to_world_space(screen_pos[3], viewport_params.xy, inv_view_proj);
    
    world_pos[4] = screen_to_world_space(screen_pos[4], viewport_params.xy, inv_view_proj);
    world_pos[5] = screen_to_world_space(screen_pos[5], viewport_params.xy, inv_view_proj);
    world_pos[6] = screen_to_world_space(screen_pos[6], viewport_params.xy, inv_view_proj);
    world_pos[7] = screen_to_world_space(screen_pos[7], viewport_params.xy, inv_view_proj);

    Frustum frustum;

    frustum.planes[0] = compute_plane(view_pos.xyz, world_pos[2].xyz, world_pos[0].xyz);     // Left
    frustum.planes[1] = compute_plane(view_pos.xyz, world_pos[1].xyz, world_pos[3].xyz);     // Right
    frustum.planes[2] = compute_plane(view_pos.xyz, world_pos[0].xyz, world_pos[1].xyz);     // Top
    frustum.planes[3] = compute_plane(view_pos.xyz, world_pos[3].xyz, world_pos[2].xyz);     // Bottom
    frustum.planes[4] = compute_plane(world_pos[3].xyz, world_pos[1].xyz, world_pos[0].xyz); // Far
    frustum.planes[5] = compute_plane(world_pos[6].xyz, world_pos[4].xyz, world_pos[5].xyz); // Near

    return frustum;
}

// ------------------------------------------------------------------

bool is_point_light_visible(uint idx, in Frustum frustum)
{
    for (int i = 0; i < 6; i++) 
    {
		vec3 normal = frustum.planes[i].xyz;
		float dist = frustum.planes[i].w;
		float side = dot(point_light_position[idx].xyz, normal) - dist;

		if (side < -point_light_near_far[idx].y)
			return false;
	}

    return true;
}

// ------------------------------------------------------------------

bool is_spot_light_visible(uint idx, in Frustum frustum)
{
    return false;
}

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
    const uint tile_idx = (gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x);

    if (gl_LocalInvocationIndex == 0)
    {
        g_MinDepth = 0xFFFFFFFF;
        g_MaxDepth = 0;
        g_PointLightCount = 0;
        g_SpotLightCount = 0;
    }

    barrier();

    const ivec2 size = textureSize(s_Depth, 0);
	const vec2 tex_coord = vec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y) / vec2(size.x - 1, size.y - 1);

    float depth = texture(s_Depth, tex_coord).r;
    
    uint depth_int = floatBitsToUint(depth);
	atomicMin(g_MinDepth, depth_int);
	atomicMax(g_MaxDepth, depth_int);

    barrier();

    if (gl_LocalInvocationIndex == 0)
        g_Frustum = create_frustum(gl_WorkGroupID.xy, g_MinDepth, g_MaxDepth);

    barrier();

    for (uint i = gl_LocalInvocationIndex; g_PointLightCount < MAX_POINT_LIGHTS_PER_TILE && i < point_light_count; i += (TILE_SIZE * TILE_SIZE))
    {
        if (is_point_light_visible(i, g_Frustum))
        {
            const uint idx = atomicAdd(g_PointLightCount, 1);

            if (idx >= MAX_POINT_LIGHTS_PER_TILE)
                break;

            indices[tile_idx].point_light_indices[idx] = i;
        }
    }

    for (uint i = gl_LocalInvocationIndex; g_SpotLightCount < MAX_SPOT_LIGHTS_PER_TILE && i < spot_light_count; i += (TILE_SIZE * TILE_SIZE))
    {
        if (is_spot_light_visible(i, g_Frustum))
        {
            const uint idx = atomicAdd(g_SpotLightCount, 1);

            if (idx >= MAX_SPOT_LIGHTS_PER_TILE)
                break;

            indices[tile_idx].spot_light_indices[idx] = i;
        }
    }

    barrier();

    if (gl_LocalInvocationIndex == 0)
    {
        indices[tile_idx].num_point_lights = min(g_PointLightCount, MAX_POINT_LIGHTS_PER_TILE);
        indices[tile_idx].num_spot_lights = min(g_SpotLightCount, MAX_SPOT_LIGHTS_PER_TILE);
    }
}

// ------------------------------------------------------------------