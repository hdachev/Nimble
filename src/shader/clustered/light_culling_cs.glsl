#include <../common/uniforms.glsl>
#include <../common/helper.glsl>
#include <../tiled/common.glsl>

#define BLOCK_SIZE 16

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

layout (local_size_x = BLOCK_SIZE, local_size_y = BLOCK_SIZE, local_size_z = 1) in;

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

vec3 closest_point(in ClusteredAABB aabb, in vec3 point)
{
    vec3 result = point;

    result.x = (result.x < aabb.aabb_min.x) ? aabb.aabb_min.x : result.x;
	result.y = (result.y < aabb.aabb_min.x) ? aabb.aabb_min.y : result.y;
	result.z = (result.z < aabb.aabb_min.x) ? aabb.aabb_min.z : result.z;

	result.x = (result.x > aabb.aabb_max.x) ? aabb.aabb_max.x : result.x;
	result.y = (result.y > aabb.aabb_max.x) ? aabb.aabb_max.y : result.y;
	result.z = (result.z > aabb.aabb_max.x) ? aabb.aabb_max.z : result.z;

	return result;
}

// ------------------------------------------------------------------

bool sphere_intersects_aabb(in Sphere sphere, in ClusteredAABB aabb) 
{
	vec3 closest_point = closest_point(aabb, sphere.c);
	float dist_sq = pow(length(sphere.c - closest_point), 2.0);
	float radius_sq = sphere.r * sphere.r;
	return dist_sq < radius_sq;
}

// ------------------------------------------------------------------

bool is_point_light_visible(uint idx, in ClusteredAABB aabb)
{
    vec4 position_vs = view_mat * vec4(point_light_position(idx), 1.0);

    Sphere sphere;

    sphere.c = position_vs.xyz;
    sphere.r = point_light_far_field(idx);

    return sphere_intersects_aabb(sphere, aabb);
}

// ------------------------------------------------------------------

bool is_spot_light_visible(uint idx, in ClusteredAABB aabb)
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
        g_PointLightCount = 0;
        g_SpotLightCount = 0;
        g_LightCount = 0;
        g_Cluster = clusters[cluster_idx];
    }

    barrier();

    for (uint i = (point_light_offset() + gl_LocalInvocationIndex); g_LightCount < MAX_LIGHTS_PER_TILE && i < point_light_count(); i += (BLOCK_SIZE * BLOCK_SIZE))
    {
        if (is_point_light_visible(i, g_Cluster))
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

    for (uint i = (spot_light_offset() + gl_LocalInvocationIndex); g_LightCount < MAX_LIGHTS_PER_TILE && i < spot_light_count(); i += (BLOCK_SIZE * BLOCK_SIZE))
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

    for (uint i = gl_LocalInvocationIndex; i < g_PointLightCount; i += (BLOCK_SIZE * BLOCK_SIZE))
        light_indices[g_PointLightStartOffset + i] = g_SharedLightList[i];

    for (uint i = gl_LocalInvocationIndex; i < g_SpotLightCount; i += (BLOCK_SIZE * BLOCK_SIZE))
        light_indices[g_SpotLightStartOffset + i] = g_SharedLightList[i];
}

// ------------------------------------------------------------------