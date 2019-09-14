#include <../../common/uniforms.glsl>
#include <../../common/helper.glsl>

// ------------------------------------------------------------------
// OUTPUTS ----------------------------------------------------------
// ------------------------------------------------------------------

out vec3 FS_OUT_FragColor;

// ------------------------------------------------------------------
// INPUTS -----------------------------------------------------------
// ------------------------------------------------------------------

in vec2 FS_IN_TexCoord;

// ------------------------------------------------------------------
// UNIFORMS ---------------------------------------------------------
// ------------------------------------------------------------------

uniform float u_Strength;
uniform sampler2D s_Color;

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
	vec3 color = texture2D(s_Color, FS_IN_TexCoord).rgb;
	
    float amount = 0.0;
    float size = 0.001;

	amount = (1.0 + size*6.0) * 0.5;
	amount *= 1.0 + size*16.0 * 0.5;
	amount *= 1.0 + size*19.0 * 0.5;
	amount *= 1.0 + size*27.0 * 0.5;
	amount = pow(amount, 3.0);

	amount *= u_Strength;
	
    vec3 col;
    col.r = texture(s_Color, vec2(FS_IN_TexCoord.x + amount, FS_IN_TexCoord.y)).r;
    col.g = texture(s_Color, FS_IN_TexCoord).g;
    col.b = texture(s_Color, vec2(FS_IN_TexCoord.x - amount, FS_IN_TexCoord.y)).b;

	col *= (1.0 - amount * 0.5);

	FS_OUT_FragColor = col;
}

// ------------------------------------------------------------------