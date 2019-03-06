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

uniform sampler2D s_Color;
uniform sampler2D s_Bloom;
uniform float u_Strength;

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
	vec3 color = texture(s_Color, FS_IN_TexCoord).rgb;

	if (u_Strength > 0)
	{
		// Fetch samples from bloom textures
		vec3 bloom = texture(s_Bloom, FS_IN_TexCoord).rgb;
		color += bloom;
	}

	FS_OUT_FragColor = color;
}

// ------------------------------------------------------------------