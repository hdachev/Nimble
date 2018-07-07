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

uniform sampler2D s_Color;
uniform sampler2D s_Bloom2;
uniform sampler2D s_Bloom4;
uniform sampler2D s_Bloom8;
uniform sampler2D s_Bloom16;
uniform float	  u_Strength;

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
	vec3 color = texture(s_Color, PS_IN_TexCoord).rgb;

	if (u_Strength > 0)
	{
		// Fetch samples from bloom textures
		vec3 bloom2 = texture(s_Bloom2, PS_IN_TexCoord).rgb;
		vec3 bloom4 = texture(s_Bloom4, PS_IN_TexCoord).rgb;
		vec3 bloom8 = texture(s_Bloom8, PS_IN_TexCoord).rgb;
		vec3 bloom16 = texture(s_Bloom16, PS_IN_TexCoord).rgb;

		// Mix values
		bloom8 = mix(bloom8, bloom16, 0.5);
		bloom4 = mix(bloom4, bloom8, 0.5);
		vec3 bloom = mix(bloom2, bloom4, 0.5);
		color = color + bloom * u_Strength;
	}

	PS_OUT_FragColor = color;
}

// ------------------------------------------------------------------