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

// ------------------------------------------------------------------
// SHARED DATA ------------------------------------------------------
// ------------------------------------------------------------------

shared uint g_MinDepth;
shared uint g_MaxDepth;

// ------------------------------------------------------------------
// FUNCTIONS --------------------------------------------------------
// ------------------------------------------------------------------

Frustum create_frustum(ivec2 idx, float min_depth, float max_depth)
{
    vec2 tile_size_ndc = 2.0 * vec2(TILE_SIZE, TILE_SIZE) / vec2(float(viewport_width), float(viewport_height));

    Frustum f;

    vec2 ndc_pos[4];

    vec2 upper_left = vec2(-1.0, 1.0);

    ndc_pos[0] = upper_left + tile_size_ndc * idx; // top left
    ndc_pos[1] = vec2(ndc_pos[0].x + tile_size_ndc.x, ndc_pos[0].y); // top right
    ndc_pos[2] = ndc_pos[0] + tile_size_ndc; // bottom right
    ndc_pos[3] = vec2(ndc_pos[0].x, ndc_pos[0].y + tile_size_ndc.y); // bottom left

    for (int i = 0; i < 4; i++)
    {
        vec4 temp = inv_view_proj * vec4(ndc_pos[i], min_depth, 1.0);
        f.points[i] = temp / temp.w;

        temp = inv_view_proj * vec4(ndc_pos[i], max_depth, 1.0);
        f.points[i + 4] = temp / temp.w;
    }

    for (int i = 0; i < 4; i++)
    {
        vec3 plane_normal = cross(f.points[i].xyz - view_pos.xyz, f.points[i + 1].xyz - view_pos.xyz);
        plane_normal = normalize(plane_normal);
        f.planes[i] = vec4(plane_normal, -dot(plane_normal, f.points[i]));
    }
    
    // near plane
    vec3 plane_normal = cross(f.points[1].xyz - f.points[0].xyz, f.points[3].xyz - f.points[0].xyz);
    plane_normal = normalize(plane_normal);
    f.planes[4] = vec4(plane_normal, -dot(plane_normal, f.points[0]));

    // far plane
    plane_normal = cross(f.points[7].xyz - f.points[4].xyz, f.points[5].xyz - f.points[4].xyz);
    plane_normal = normalize(plane_normal);
    f.planes[5] = vec4(plane_normal, -dot(plane_normal, f.points[4]));

    return f;
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

    float depth = imageLoad(i_Depth, gl_GlobalInvocationID.xy).r;
    depth = linear_01_depth(depth)l

    uint depth_int = floatBitsToUint(depth);
	atomicMin(g_MinDepth, depth_int);
	atomicMax(g_MaxDepth, depth_int);

    barrier();

    if (gl_LocalInvocationID.x == 0 && gl_LocalInvocationID.y == 0)
        frustums[tile_idx] = create_frustum(gl_WorkGroupID .xy, min_depth, max_depth);
}

// ------------------------------------------------------------------