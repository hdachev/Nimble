layout (location = 0) out vec3 PS_OUT_NearFillDoF4;
layout (location = 1) out vec3 PS_OUT_FarFillDoF4;

in vec2 PS_IN_TexCoord;

uniform vec2 u_PixelSize;

uniform sampler2D s_CoC4;
uniform sampler2D s_NearBlurCoC4;
uniform sampler2D s_NearDoF4;
uniform sampler2D s_FarDoF4;

// ----------------------------------------------------------------------
// Fill Bokeh Shapes.
// ----------------------------------------------------------------------

void main()
{
	float coc_near_blurred = texture(s_NearBlurCoC4, PS_IN_TexCoord).x;
	float coc_far = texture(s_CoC4, PS_IN_TexCoord).y;

	PS_OUT_NearFillDoF4 = texture(s_NearDoF4, PS_IN_TexCoord);
	PS_OUT_FarFillDoF4 = texture(s_FarDoF4, PS_IN_TexCoord);	

	if (coc_near_blurred > 0.0f)
	{
		for (int i = -1; i <= 1; i++)
		{
			for (int j = -1; j <= 1; j++)
			{
				vec2 sample_tex_coord = PS_IN_TexCoord + vec2(i, j) * u_PixelSize;
				vec4 sample = texture(s_NearDoF4, sample_tex_coord, 0);
				PS_OUT_NearFillDoF4 = max(PS_OUT_NearFillDoF4, sample);
			}
		}
	}

	if (coc_far > 0.0f)
	{
		for (int i = -1; i <= 1; i++)
		{
			for (int j = -1; j <= 1; j++)
			{
				vec2 sample_tex_coord = PS_IN_TexCoord + vec2(i, j) * u_PixelSize;
				vec4 sample = texture(s_FarDoF4, sample_tex_coord, 0);
				PS_OUT_FarFillDoF4 = max(PS_OUT_FarFillDoF4, sample);
			}
		}
	}
}

// ----------------------------------------------------------------------