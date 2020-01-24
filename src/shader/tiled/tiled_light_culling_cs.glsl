#include <../common/uniforms.glsl>

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

layout(std430, binding = 0) buffer u_TileFrustums
{
	Frustum frustums[];
};

layout(std430, binding = 1) buffer u_LightIndices
{
	LightIndices indices[];
};

// ------------------------------------------------------------------
// SHARED -----------------------------------------------------------
// ------------------------------------------------------------------

shared float g_MinDepth;
shared float g_MaxDepth;

// ------------------------------------------------------------------
// FUNCTIONS --------------------------------------------------------
// ------------------------------------------------------------------

bool is_point_light_visible(uint idx, in Frustum frustum)
{
    for (int i = 0; i < 6; i++) 
    {
		vec3 normal = frustum.planes[i].xyz;
		float dist = frustum.planes[i].w;
		float side = dot(point_light_position[idx].xyz, normal) + dist;

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
    uint tile_idx = (gl_LocalInvocationID.y * TILE_SIZE + gl_LocalInvocationID.x);

    Frustum frustum = frustums[tile_idx];

    uint point_lights_per_thread = MAX_POINT_LIGHTS / (TILE_SIZE * TILE_SIZE);
    uint spot_lights_per_thread = MAX_SPOT_LIGHTS / (TILE_SIZE * TILE_SIZE);

    uint start_idx = tile_idx * point_lights_per_thread;

    for (uint i = start_idx; i < (start_idx + point_lights_per_thread); i++)
    {
        if (is_point_light_visible(i, frustum))
        {
            uint idx = atomicAdd(indices[tile_idx].num_point_lights, 1);
            indices[tile_idx].point_light_indices[idx] = i;
        }
    }

    start_idx = tile_idx * spot_lights_per_thread;

    for (uint i = start_idx; i < (start_idx + spot_lights_per_thread); i++)
    {
        if (is_spot_light_visible(i, frustum))
        {
            uint idx = atomicAdd(indices[tile_idx].num_spot_lights, 1);
            indices[tile_idx].spot_light_indices[idx] = i;
        }
    }
}

// ------------------------------------------------------------------