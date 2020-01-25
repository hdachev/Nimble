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

// ------------------------------------------------------------------
// FUNCTIONS --------------------------------------------------------
// ------------------------------------------------------------------

Frustum create_frustum(uvec2 idx, uint min_depth, uint max_depth)
{
    vec2 tile_size_ndc = 2.0 * vec2(TILE_SIZE, TILE_SIZE) / vec2(float(viewport_width), float(viewport_height));

    Frustum f;

    vec2 ndc_pos[4];

    vec2 upper_left = vec2(-1.0, 1.0);

    ndc_pos[0] = upper_left + tile_size_ndc * idx; // top left
    ndc_pos[1] = vec2(ndc_pos[0].x + tile_size_ndc.x, ndc_pos[0].y); // top right
    ndc_pos[2] = ndc_pos[0] + tile_size_ndc; // bottom right
    ndc_pos[3] = vec2(ndc_pos[0].x, ndc_pos[0].y + tile_size_ndc.y); // bottom left

    float fmin_depth = uintBitsToFloat(min_depth);
    float fmax_depth = uintBitsToFloat(max_depth);

    for (int i = 0; i < 4; i++)
    {
        vec4 temp = inv_view_proj * vec4(ndc_pos[i], fmin_depth, 1.0);
        f.points[i] = temp / temp.w;

        temp = inv_view_proj * vec4(ndc_pos[i], fmax_depth, 1.0);
        f.points[i + 4] = temp / temp.w;
    }

    for (int i = 0; i < 4; i++)
    {
        vec3 plane_normal = cross(f.points[i].xyz - view_pos.xyz, f.points[i + 1].xyz - view_pos.xyz);
        plane_normal = normalize(plane_normal);
        f.planes[i] = vec4(plane_normal, -dot(plane_normal, f.points[i].xyz));
    }
    
    // near plane
    vec3 plane_normal = cross(f.points[1].xyz - f.points[0].xyz, f.points[3].xyz - f.points[0].xyz);
    plane_normal = normalize(plane_normal);
    f.planes[4] = vec4(plane_normal, -dot(plane_normal, f.points[0].xyz));

    // far plane
    plane_normal = cross(f.points[7].xyz - f.points[4].xyz, f.points[5].xyz - f.points[4].xyz);
    plane_normal = normalize(plane_normal);
    f.planes[5] = vec4(plane_normal, -dot(plane_normal, f.points[4].xyz));

    return f;
}

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

    if (gl_LocalInvocationIndex == 0)
    {
        g_MinDepth = 0xFFFFFFFF;
        g_MaxDepth = 0;
    }

    barrier();

    ivec2 size = textureSize(s_Depth, 0);
	vec2 tex_coord = vec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y) / vec2(size.x - 1, size.y - 1);

    float depth = texture(s_Depth, tex_coord).r;
    
    // Linearize                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
    depth = linear_01_depth(depth);

    uint depth_int = floatBitsToUint(depth);
	atomicMin(g_MinDepth, depth_int);
	atomicMax(g_MaxDepth, depth_int);

    barrier();

    if (gl_LocalInvocationID.x == 0 && gl_LocalInvocationID.y == 0)
        g_Frustum = create_frustum(gl_WorkGroupID.xy, g_MinDepth, g_MaxDepth);

    barrier();

    uint point_lights_per_thread = MAX_POINT_LIGHTS / (TILE_SIZE * TILE_SIZE);
    uint spot_lights_per_thread = MAX_SPOT_LIGHTS / (TILE_SIZE * TILE_SIZE);

    uint start_idx = tile_idx * point_lights_per_thread;

    for (uint i = start_idx; i < (start_idx + point_lights_per_thread); i++)
    {
        if (is_point_light_visible(i, g_Frustum))
        {
            uint idx = atomicAdd(indices[tile_idx].num_point_lights, 1);
            indices[tile_idx].point_light_indices[idx] = i;
        }
    }

    start_idx = tile_idx * spot_lights_per_thread;

    for (uint i = start_idx; i < (start_idx + spot_lights_per_thread); i++)
    {
        if (is_spot_light_visible(i, g_Frustum))
        {
            uint idx = atomicAdd(indices[tile_idx].num_spot_lights, 1);
            indices[tile_idx].spot_light_indices[idx] = i;
        }
    }
}

// ------------------------------------------------------------------