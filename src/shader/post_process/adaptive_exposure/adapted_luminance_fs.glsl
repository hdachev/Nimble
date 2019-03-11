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

uniform sampler2D s_PreviousLuminance;
uniform sampler2D s_CurrentLuminance;

uniform float u_Delta;
uniform float u_Tau;

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------


void main(void)
{
    float prev_lum = texture(s_PreviousLuminance, FS_IN_TexCoord).r;
    float curr_lum = texture(s_CurrentLuminance, FS_IN_TexCoord).r;

    // Adapt the luminance using Pattanaik's technique    
    float adapted_lum = prev_lum + (curr_lum - curr_lum) * (1 - exp(-u_Delta * u_Tau));

    FS_OUT_Color = vec4(log(adapted_lum), 0.0, 0.0, 0.0);
}

// ------------------------------------------------------------------