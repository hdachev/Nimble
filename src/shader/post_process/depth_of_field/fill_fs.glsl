// ------------------------------------------------------------------
// INPUT VARIABLES  -------------------------------------------------
// ------------------------------------------------------------------

in vec2 FS_IN_TexCoord;

// ------------------------------------------------------------------
// OUTPUT VARIABLES  ------------------------------------------------
// ------------------------------------------------------------------

layout (location = 0) out vec3 FS_OUT_NearFillDoF4;
layout (location = 1) out vec3 FS_OUT_FarFillDoF4;

// ------------------------------------------------------------------
// UNIFORMS  --------------------------------------------------------
// ------------------------------------------------------------------

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
	float coc_near_blurred = texture(s_NearBlurCoC4, FS_IN_TexCoord).x;
	float coc_far = texture(s_CoC4, FS_IN_TexCoord).y;

	FS_OUT_NearFillDoF4 = texture(s_NearDoF4, FS_IN_TexCoord).xyz;
	FS_OUT_FarFillDoF4 = texture(s_FarDoF4, FS_IN_TexCoord).xyz;

	if (coc_near_blurred > 0.0)
	{
		for (int i = -1; i <= 1; i++)
		{
			for (int j = -1; j <= 1; j++)
			{
				vec2 sample_tex_coord = FS_IN_TexCoord + vec2(i, j) * u_PixelSize;
				vec3 dof_sample = texture(s_NearDoF4, sample_tex_coord).xyz;
				FS_OUT_NearFillDoF4 = max(FS_OUT_NearFillDoF4, dof_sample);
			}
		}
	}

	if (coc_far > 0.0)
	{
		for (int i = -1; i <= 1; i++)
		{
			for (int j = -1; j <= 1; j++)
			{
				vec2 sample_tex_coord = FS_IN_TexCoord + vec2(i, j) * u_PixelSize;
				vec3 dof_sample = texture(s_FarDoF4, sample_tex_coord).xyz;
				FS_OUT_FarFillDoF4 = max(FS_OUT_FarFillDoF4, dof_sample);
			}
		}
	}
}

// ----------------------------------------------------------------------