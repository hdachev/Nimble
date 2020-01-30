#include <../common/uniforms.glsl>
#include <../common/helper.glsl>
#include <common.glsl>

// ------------------------------------------------------------------
// DEFINES ----------------------------------------------------------
// ------------------------------------------------------------------

#define TILE_SIZE 16
#define MAX_POINT_LIGHTS_PER_TILE 512
#define MAX_SPOT_LIGHTS_PER_TILE 512

// ------------------------------------------------------------------
// STRUCTURES -------------------------------------------------------
// ------------------------------------------------------------------

struct Plane
{
    vec3 N;   // Plane normal.
    float  d; // Distance to origin.
};

struct Sphere
{
    vec3 c;   // Center point.
    float  r; // Radius.
};

// ------------------------------------------------------------------
// INPUTS -----------------------------------------------------------
// ------------------------------------------------------------------

layout (local_size_x = TILE_SIZE, local_size_y = TILE_SIZE, local_size_z = 1) in;

// ------------------------------------------------------------------
// UNIFORMS ---------------------------------------------------------
// ------------------------------------------------------------------

layout (std430, binding = 3) buffer u_Frustums
{
	Frustum frustums[];
};

layout(std430, binding = 4) buffer u_LightIndices
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

bool sphere_inside_plane(Sphere sphere, Plane plane)
{
    return dot(plane.N, sphere.c) - plane.d < -sphere.r;
}

// ------------------------------------------------------------------

bool sphere_inside_frustum(Sphere sphere, Frustum frustum, float zNear, float zFar)
{
    bool result = true;

    // First check depth
    // Note: Here, the view vector points in the -Z axis so the 
    // far depth value will be approaching -infinity.
    if ( (sphere.c.z - sphere.r) > zNear || (sphere.c.z + sphere.r) < zFar )
        result = false;

    // Then check frustum planes
    for (int i = 0; i < 4 && result; i++)
    {
        Plane p;

        p.N = frustum.planes[i].xyz;
        p.d = frustum.planes[i].w;

        if (sphere_inside_plane( sphere, p))
            result = false;
    }

    return result;
}

// ------------------------------------------------------------------

bool is_point_light_visible(uint idx, in Frustum frustum, float zNear, float zFar)
{
    vec4 position_vs = view_mat * vec4(point_light_position[idx].xyz, 1.0);

    Sphere sphere;

    sphere.c = position_vs.xyz;
    sphere.r = point_light_near_far[idx].y;

    return sphere_inside_frustum(sphere, frustum, zNear, zFar);
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
        g_Frustum = frustums[tile_idx];
    }

    barrier();

    float depth = 2.0 * texelFetch(s_Depth, ivec2(gl_GlobalInvocationID.xy), 0).r - 1.0;
    
    uint depth_int = floatBitsToUint(depth);
	atomicMin(g_MinDepth, depth_int);
	atomicMax(g_MaxDepth, depth_int);

    barrier();

    float fmin_depth = uintBitsToFloat(g_MinDepth);
    float fmax_depth = uintBitsToFloat(g_MaxDepth);

    float min_depth_vs = clip_to_view_space(vec4(0.0, 0.0, fmin_depth, 1.0), inv_proj).z;
    float max_depth_vs = clip_to_view_space(vec4(0.0, 0.0, fmax_depth, 1.0), inv_proj).z;

    barrier();

    for (uint i = gl_LocalInvocationIndex; g_PointLightCount < MAX_POINT_LIGHTS_PER_TILE && i < point_light_count; i += (TILE_SIZE * TILE_SIZE))
    {
        if (is_point_light_visible(i, g_Frustum, min_depth_vs, max_depth_vs))
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