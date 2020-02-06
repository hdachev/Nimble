#include <../common/uniforms.glsl>
#include <../common/helper.glsl>
#include <../tiled/common.glsl>

// ------------------------------------------------------------------
// INPUTS -----------------------------------------------------------
// ------------------------------------------------------------------

layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

// ------------------------------------------------------------------
// UNIFORMS ---------------------------------------------------------
// ------------------------------------------------------------------

layout (std430, binding = 3) buffer u_ClusterAABBs
{
	ClusteredAABB clusters[];
};

// ------------------------------------------------------------------
// FUNCTIONS --------------------------------------------------------
// ------------------------------------------------------------------

vec3 line_intersection_to_z_plane(vec3 A, vec3 B, float z)
{
    //Because this is a Z based normal this is fixed
    vec3 normal = vec3(0.0, 0.0, 1.0);

    vec3 ab =  B - A;

    //Computing the intersection length for the line and the plane
    float t = (z - dot(normal, A)) / dot(normal, ab);

    //Computing the actual xyz position of the point along the line
    vec3 result = A + t * ab;

    return result;
}

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
    const uint cluster_idx = gl_WorkGroupID.x +
                             gl_WorkGroupID.y * gl_NumWorkGroups.x +
                             gl_WorkGroupID.z * (gl_NumWorkGroups.x * gl_NumWorkGroups.y);

    vec3 camera_pos_vs = vec3(0.0);

    vec4 max_point_ss = vec4((gl_WorkGroupID.x + 1) * TILE_SIZE, (gl_WorkGroupID.y + 1) * TILE_SIZE, 1, 1);
    vec4 min_point_ss = vec4(gl_WorkGroupID.xy * TILE_SIZE, -1, 1); 

    vec4 max_point_vs = screen_to_view_space(max_point_ss, viewport_params.xy, inv_proj);
    vec4 min_point_vs = screen_to_view_space(min_point_ss, viewport_params.xy, inv_proj);

    float tile_near = -z_buffer_params.x * pow(z_buffer_params.y / z_buffer_params.x, gl_WorkGroupID.z / float(gl_NumWorkGroups.z));
    float tile_far  = -z_buffer_params.x * pow(z_buffer_params.y / z_buffer_params.x, (gl_WorkGroupID.z + 1) / float(gl_NumWorkGroups.z));

    vec3 min_point_near = line_intersection_to_z_plane(camera_pos_vs, min_point_vs.xyz, tile_near);
    vec3 min_point_far  = line_intersection_to_z_plane(camera_pos_vs, min_point_vs.xyz, tile_far);
    vec3 max_point_near = line_intersection_to_z_plane(camera_pos_vs, max_point_vs.xyz, tile_near);
    vec3 max_point_far  = line_intersection_to_z_plane(camera_pos_vs, max_point_vs.xyz, tile_far);

    vec3 aabb_min = min(min(min_point_near, min_point_far),min(max_point_near, max_point_far));
    vec3 aabb_max = max(max(min_point_near, min_point_far),max(max_point_near, max_point_far));

    clusters[cluster_idx].aabb_min  = vec4(aabb_min , 0.0);
    clusters[cluster_idx].aabb_max  = vec4(aabb_max , 0.0);
}

// ------------------------------------------------------------------