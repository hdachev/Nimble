#include <../common/gaussian_blur.glsl>

// ------------------------------------------------------------------
// OUTPUTS ----------------------------------------------------------
// ------------------------------------------------------------------

out vec3 PS_OUT_FragColor;

// ------------------------------------------------------------------
// INPUTS -----------------------------------------------------------
// ------------------------------------------------------------------

in vec2 PS_IN_TexCoord;

// ------------------------------------------------------------------
// UNIFORMS ---------------------------------------------------------
// ------------------------------------------------------------------

uniform sampler2D s_Texture;
uniform vec2 u_Direction;
uniform vec2 u_Resolution;

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
#ifdef GAUSSIAN_BLUR_5x5
	vec4 result = gaussian_blur_5x5(s_Texture, PS_IN_TexCoord, u_Resolution, u_Direction);
#elif GAUSSIAN_BLUR_13x13
	vec4 result = gaussian_blur_13x13(s_Texture, PS_IN_TexCoord, u_Resolution, u_Direction);
#else
	vec4 result = gaussian_blur_9x9(s_Texture, PS_IN_TexCoord, u_Resolution, u_Direction);
#endif
	PS_OUT_FragColor = result.rgb;
}

// ------------------------------------------------------------------