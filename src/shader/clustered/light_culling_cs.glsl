#include <../common/uniforms.glsl>
#include <../common/helper.glsl>
#include <../tiled/common.glsl>

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

layout (local_size_x = TILE_SIZE, local_size_y = TILE_SIZE, local_size_z = TILE_SIZE) in;

// ------------------------------------------------------------------
// UNIFORMS ---------------------------------------------------------
// ------------------------------------------------------------------

layout (std430, binding = 3) buffer u_ClusterAABBs
{
	ClusteredAABB clusters[];
};

layout(std430, binding = 4) buffer u_LightIndices
{
	uint light_indices[];
};

layout(std430, binding = 5) buffer u_LightGrid
{
	uvec4 light_grid[];
};

layout(std430, binding = 6) buffer u_LightCounter
{
	uvec4 light_counter;
};

// ------------------------------------------------------------------
// SHARED DATA ------------------------------------------------------
// ------------------------------------------------------------------

shared uint g_MinDepth;
shared uint g_MaxDepth;
shared ClusteredAABB g_Cluster;
shared uint g_PointLightCount;
shared uint g_SpotLightCount;
shared uint g_PointLightStartOffset;
shared uint g_SpotLightStartOffset;
shared uint g_LightCount;
shared uint g_SharedLightList[MAX_LIGHTS_PER_CLUSTER];

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
    vec4 position_vs = view_mat * vec4(point_light_position(idx), 1.0);

    Sphere sphere;

    sphere.c = position_vs.xyz;
    sphere.r = point_light_far_field(idx);

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
    const uint cluster_idx = gl_WorkGroupID.x +
                             gl_WorkGroupID.y * gl_NumWorkGroups.x +
                             gl_WorkGroupID.z * (gl_NumWorkGroups.x * gl_NumWorkGroups.y);

    if (gl_LocalInvocationIndex == 0)
    {
        g_MinDepth = 0xFFFFFFFF;
        g_MaxDepth = 0;
        g_PointLightCount = 0;
        g_SpotLightCount = 0;
        g_LightCount = 0;
        g_Cluster = clusters[cluster_idx];
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

    for (uint i = (point_light_offset() + gl_LocalInvocationIndex); g_LightCount < MAX_LIGHTS_PER_TILE && i < point_light_count(); i += (TILE_SIZE * TILE_SIZE * TILE_SIZE))
    {
        if (is_point_light_visible(i, g_Cluster, min_depth_vs, max_depth_vs))
        {
            const uint idx = atomicAdd(g_LightCount, 1);

            if (idx >= MAX_LIGHTS_PER_TILE)
                break;

            g_SharedLightList[idx] = i;
        }
    }

    barrier();

    if (gl_LocalInvocationIndex == 0)
        g_PointLightCount = g_LightCount;

    for (uint i = (spot_light_offset() + gl_LocalInvocationIndex); g_LightCount < MAX_LIGHTS_PER_TILE && i < spot_light_count(); i += (TILE_SIZE * TILE_SIZE * TILE_SIZE))
    {
        if (is_spot_light_visible(i, g_Cluster))
        {
            const uint idx = atomicAdd(g_LightCount, 1);

            if (idx >= MAX_LIGHTS_PER_TILE)
                break;

           g_SharedLightList[idx] = i;
        }
    }

    barrier();

    if (gl_LocalInvocationIndex == 0)
    {
        g_SpotLightCount = g_LightCount - g_PointLightCount;

        light_grid[cluster_idx].x = g_PointLightCount;
        light_grid[cluster_idx].y = g_SpotLightCount;

        g_PointLightStartOffset = atomicAdd(light_counter.x, g_PointLightCount);
        g_SpotLightStartOffset = atomicAdd(light_counter.y, g_SpotLightCount);

        light_grid[cluster_idx].z = g_PointLightStartOffset;
        light_grid[cluster_idx].w = g_SpotLightStartOffset;
    }

    barrier();

    for (uint i = gl_LocalInvocationIndex; i < g_PointLightCount; i += (TILE_SIZE * TILE_SIZE * TILE_SIZE))
        light_indices[g_PointLightStartOffset + i] = g_SharedLightList[i];

    for (uint i = gl_LocalInvocationIndex; i < g_SpotLightCount; i += (TILE_SIZE * TILE_SIZE * TILE_SIZE))
        light_indices[g_SpotLightStartOffset + i] = g_SharedLightList[i];
}

// ------------------------------------------------------------------