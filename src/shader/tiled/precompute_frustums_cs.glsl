#include <../common/uniforms.glsl>
#include <../common/helper.glsl>
#include <common.glsl>

// ------------------------------------------------------------------
// INPUTS -----------------------------------------------------------
// ------------------------------------------------------------------

layout (local_size_x = TILE_SIZE, local_size_y = TILE_SIZE, local_size_z = 1) in;

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
 
    screen_pos[0] = vec4(gl_WorkGroupID.xy * TILE_SIZE, -1.0, 1.0);                               // Top-Left
    screen_pos[1] = vec4((gl_WorkGroupID.x + 1, gl_WorkGroupID.y) * TILE_SIZE, -1.0, 1.0f;     // Top-Right
    screen_pos[2] = vec4((gl_WorkGroupID.x, gl_WorkGroupID.y + 1)  * TILE_SIZE, -1.0, 1.0);    // Bottom-Left
    screen_pos[3] = vec4((gl_WorkGroupID.x + 1, gl_WorkGroupID.y + 1) * TILE_SIZE, -1.0, 1.0); // Bottom-Right
 
    vec4 world_pos[4];
 
    world_pos[0] = screen_to_world_space(screen_pos[0], viewport_params.xy, inv_view_proj);
    world_pos[1] = screen_to_world_space(screen_pos[1], viewport_params.xy, inv_view_proj);
    world_pos[2] = screen_to_world_space(screen_pos[2], viewport_params.xy, inv_view_proj);
    world_pos[3] = screen_to_world_space(screen_pos[3], viewport_params.xy, inv_view_proj);
 
    Frustum frustum;
 
    frustum.planes[0] = compute_plane(view_pos.xyz, world_pos[2], world_pos[0]); // Left
    frustum.planes[1] = compute_plane(view_pos.xyz, world_pos[1], world_pos[3]); // Right
    frustum.planes[2] = compute_plane(view_pos.xyz, world_pos[0], world_pos[1]); // Top
    frustum.planes[3] = compute_plane(view_pos.xyz, world_pos[3], world_pos[2]); // Bottom
 
    if (gl_GlobalInvocationID.x < u_TileCountX || gl_GlobalInvocationID.y < u_TileCountY)
        frustums[tile_idx] = frustum;

}

// ------------------------------------------------------------------