#include <../common/uniforms.glsl>

// Using a fullscreen triangle for post-processing.
// https://www.saschawillems.de/?page_id=2122
// https://michaldrobot.com/2014/04/01/gcn-execution-patterns-in-full-screen-passes/

// ------------------------------------------------------------------
// OUTPUT VARIABLES  ------------------------------------------------
// ------------------------------------------------------------------

out vec3 PS_IN_TexCoord;

// ------------------------------------------------------------------
// MAIN  ------------------------------------------------------------
// ------------------------------------------------------------------

void main(void)
{
     const vec3 vertices[4] = vec3[4](vec3(-1.0f, -1.0f, 1.0f),
                                      vec3( 1.0f, -1.0f, 1.0f),
                                      vec3(-1.0f,  1.0f, 1.0f),
                                      vec3( 1.0f,  1.0f, 1.0f));


    vec4 clip_pos = vec4(vertices[gl_VertexID].xy, -1.0, 1.0);
    vec4 view_pos  = inv_view_proj * clip_pos;
   
    vec3 dir = vec3(view_pos);
    dir = normalize(dir);

    PS_IN_TexCoord = dir;

    gl_Position = vec4(vertices[gl_VertexID], 1.0f);
}

// ------------------------------------------------------------------