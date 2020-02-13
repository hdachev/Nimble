// ------------------------------------------------------------------
// DEFINES ----------------------------------------------------------
// ------------------------------------------------------------------

#define TILE_SIZE 16
#define MAX_LIGHTS_PER_CLUSTER 1024
#define MAX_LIGHTS_PER_TILE 1024
#define CLUSTER_GRID_DIM_X 16
#define CLUSTER_GRID_DIM_Y 8
#define CLUSTER_GRID_DIM_Z 24
#define CLUSTER_DEBUG_MAX_LIGHTS 200

struct Frustum
{
    vec4 planes[4];
};

struct ClusteredAABB
{
    vec4 aabb_min;
    vec4 aabb_max;
};

vec4 screen_to_view_space(vec4 p, vec2 screen_size, mat4 inv_proj)
{
    vec2 tex_coord = p.xy / screen_size;
 
    vec4 view_pos = inv_proj * vec4(2.0 * tex_coord - 1.0, p.z, p.w); 
 
    return view_pos / view_pos.w;
}

vec4 clip_to_view_space(vec4 p, mat4 inv_proj)
{
    vec4 view_pos = inv_proj * p; 
 
    return view_pos / view_pos.w;
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
