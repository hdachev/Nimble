#include <../common/uniforms.glsl>
#include <../common/helper.glsl>
#include <common.glsl>

// ------------------------------------------------------------------
// INPUTS -----------------------------------------------------------
// ------------------------------------------------------------------

layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

// ------------------------------------------------------------------
// UNIFORMS ---------------------------------------------------------
// ------------------------------------------------------------------

layout(std430, binding = 3) buffer u_Frustums
{
	Frustum frustums[];
};

uniform uint u_TileCountX;
uniform uint u_TileCountY;

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
    const uvec2 tile_id = gl_GlobalInvocationID.xy;
    const uint tile_idx = tile_id.y * u_TileCountX + tile_id.x;

    vec4 screen_pos[4];
 
    screen_pos[0] = vec4(gl_WorkGroupID.xy * TILE_SIZE, -1.0, 1.0);                                // Top-Left
    screen_pos[1] = vec4(vec2(gl_WorkGroupID.x + 1, gl_WorkGroupID.y) * TILE_SIZE, -1.0, 1.0);     // Top-Right
    screen_pos[2] = vec4(vec2(gl_WorkGroupID.x, gl_WorkGroupID.y + 1)  * TILE_SIZE, -1.0, 1.0);    // Bottom-Left
    screen_pos[3] = vec4(vec2(gl_WorkGroupID.x + 1, gl_WorkGroupID.y + 1) * TILE_SIZE, -1.0, 1.0); // Bottom-Right
 
    vec4 view_pos[4];
 
    view_pos[0] = screen_to_view_space(screen_pos[0], viewport_params.xy, inv_proj);
    view_pos[1] = screen_to_view_space(screen_pos[1], viewport_params.xy, inv_proj);
    view_pos[2] = screen_to_view_space(screen_pos[2], viewport_params.xy, inv_proj);
    view_pos[3] = screen_to_view_space(screen_pos[3], viewport_params.xy, inv_proj);
 
    vec3 camera_pos_vs = vec3(0.0); // Camera position is the origin in view-space.

    Frustum frustum;
 
    frustum.planes[0] = compute_plane(camera_pos_vs, view_pos[2].xyz, view_pos[0].xyz); // Left
    frustum.planes[1] = compute_plane(camera_pos_vs, view_pos[1].xyz, view_pos[3].xyz); // Right
    frustum.planes[2] = compute_plane(camera_pos_vs, view_pos[0].xyz, view_pos[1].xyz); // Top
    frustum.planes[3] = compute_plane(camera_pos_vs, view_pos[3].xyz, view_pos[2].xyz); // Bottom
 
    if (gl_GlobalInvocationID.x < u_TileCountX || gl_GlobalInvocationID.y < u_TileCountY)
        frustums[tile_idx] = frustum;

}

// ------------------------------------------------------------------