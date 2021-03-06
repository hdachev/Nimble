
#define DELTA 0.00000001

// ------------------------------------------------------------------
// INPUT VARIABLES  -------------------------------------------------
// ------------------------------------------------------------------

in vec2 FS_IN_TexCoord;

// ------------------------------------------------------------------
// OUTPUT VARIABLES  ------------------------------------------------
// ------------------------------------------------------------------

out vec4 FS_OUT_Color;

// ------------------------------------------------------------------
// UNIFORMS  --------------------------------------------------------
// ------------------------------------------------------------------

uniform sampler2D s_Texture;

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------


void main(void)
{
    vec3 color = texture2D(s_Texture, FS_IN_TexCoord).rgb;
    float luminance = max(dot(color, vec3(0.299, 0.587, 0.114)), 0.0001);

    FS_OUT_Color = vec4(log(luminance + DELTA), 0.0, 0.0, 0.0);
}

// ------------------------------------------------------------------