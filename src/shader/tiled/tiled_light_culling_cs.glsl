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

layout(std430, binding = 0) buffer u_LightIndices
{
	LightIndices indices[];
};

// ------------------------------------------------------------------
// SHARED -----------------------------------------------------------
// ------------------------------------------------------------------

shared float g_MinDepth;
shared float g_MaxDepth;
shared float g_TileDepth[TILE_SIZE][TILE_SIZE];
shared float g_ColumnDepthMin[TILE_SIZE];
shared float g_ColumnDepthMax[TILE_SIZE];

// ------------------------------------------------------------------
// FUNCTIONS --------------------------------------------------------
// ------------------------------------------------------------------

Frustum create_frustum(ivec2 idx)
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
        vec4 temp = inv_view_proj * vec4(ndc_pos[i], g_MinDepth, 1.0);
        f.points[i] = temp / temp.w;

        temp = inv_view_proj * vec4(ndc_pos[i], g_MaxDepth, 1.0);
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

bool is_point_light_visible(uint idx, in Frustum frustum)
{

}

// ------------------------------------------------------------------

bool is_spot_light_visible(uint idx, in Frustum frustum)
{
    
}

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
    uint tile_idx = (gl_LocalInvocationID.y * TILE_SIZE + gl_LocalInvocationID.x);

    g_TileDepth[gl_LocalInvocationID.x][gl_LocalInvocationID.y] = imageLoad(i_Depth, gl_GlobalInvocationID.xy).r;

    barrier();

    float min_depth = 1.0;
    float max_depth = 0.0;

    if (gl_LocalInvocationID.x == 0)
    {
        for (int i = 0; i < TILE_SIZE; i++)
        {
            if (g_TileDepth[i][gl_LocalInvocationID.y] < min_depth)
                min_depth = g_TileDepth[i][gl_LocalInvocationID.y];

            if (g_TileDepth[i][gl_LocalInvocationID.y] > max_depth)
                max_depth = g_TileDepth[i][gl_LocalInvocationID.y];
        }
    }

    g_ColumnDepthMin[gl_LocalInvocationID.y] = min_depth;
    g_ColumnDepthMax[gl_LocalInvocationID.y] = max_depth;

    barrier();

    g_MinDepth = 1.0;
    g_MaxDepth = 0.0;

    if (gl_LocalInvocationID.x == 0 && gl_LocalInvocationID.y == 0)
    {
        indices[tile_idx].num_point_lights = 0;
        indices[tile_idx].num_spot_lights = 0;

        for (int i = 0; i < TILE_SIZE; i++)
        {
            if (g_ColumnDepthMin[i] < g_MinDepth)
                g_MinDepth = g_ColumnDepthMin[i];

            if (g_ColumnDepthMax[i] > g_MaxDepth)
                g_MaxDepth = g_ColumnDepthMax[i];
        }
    }

    barrier();

    Frustum frustum = create_frustum(gl_WorkGroupID .xy);

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