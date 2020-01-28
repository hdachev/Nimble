#define TILE_SIZE 16
#define MAX_POINT_LIGHTS_PER_TILE 512
#define MAX_SPOT_LIGHTS_PER_TILE 512

struct LightIndices
{
    uint num_point_lights;
    uint num_spot_lights;
    uint point_light_indices[MAX_POINT_LIGHTS_PER_TILE];
    uint spot_light_indices[MAX_SPOT_LIGHTS_PER_TILE];
};

struct Frustum
{
    vec4 planes[4];
};

vec4 screen_to_world_space(vec4 p, vec2 screen_size, mat4 inv_view_proj)
{
    vec2 tex_coord = p.xy / screen_size;
 
    vec4 world_pos = inv_view_proj * vec4(2.0 * tex_coord - 1.0, p.z, p.w); 
 
    return world / world.w;
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
