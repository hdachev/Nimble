#include <../../../common/uniforms.glsl>

// ------------------------------------------------------------------
// UNIFORMS ---------------------------------------------------------
// ------------------------------------------------------------------

uniform int u_LightIdx;

// ------------------------------------------------------------------
// INPUT ------------------------------------------------------------
// ------------------------------------------------------------------

in vec3 PS_IN_FragPos;

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
    vec3 light_pos = point_light_position[u_LightIdx].xyz;
    float near_plane = point_light_near_far[u_LightIdx].x;
    float far_plane = point_light_near_far[u_LightIdx].y;

    float light_distance = length(PS_IN_FragPos - light_pos);
    
    gl_FragDepth = (light_distance - near_plane) / (far_plane - near_plane);
}

// ------------------------------------------------------------------