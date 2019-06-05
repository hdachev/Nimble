// ------------------------------------------------------------------
// INPUT VARIABLES  -------------------------------------------------
// ------------------------------------------------------------------

in vec2 FS_IN_TexCoord;

// ------------------------------------------------------------------
// OUTPUT VARIABLES  ------------------------------------------------
// ------------------------------------------------------------------

out vec3 FS_OUT_FragColor;

// ------------------------------------------------------------------
// UNIFORMS  --------------------------------------------------------
// ------------------------------------------------------------------

uniform vec2 u_PixelSize;
uniform float u_Blend;

uniform sampler2D s_Color;
uniform sampler2D s_CoC;
uniform sampler2D s_CoC4;
uniform sampler2D s_CoCBlur4;
uniform sampler2D s_NearDoF4;
uniform sampler2D s_FarDoF4;

// ----------------------------------------------------------------------
// Combine all results.
// ----------------------------------------------------------------------

void main()
{
    vec3 result = texture(s_Color, FS_IN_TexCoord).xyz;

	// Far Field
	{
		vec2 tex_coord_00 = FS_IN_TexCoord;
		vec2 tex_coord_10 = FS_IN_TexCoord + vec2(u_PixelSize.x, 0.0);
		vec2 tex_coord_01 = FS_IN_TexCoord + vec2(0.0, u_PixelSize.y);
		vec2 tex_coord_11 = FS_IN_TexCoord + vec2(u_PixelSize.x, u_PixelSize.y);

		float coc_far = texture(s_CoC, FS_IN_TexCoord).y;
		vec4 cocs_far_x4 = textureGather(s_CoC4, tex_coord_00, 1).wzxy;
		vec4 cocs_far_diffs = abs(vec4(coc_far) - cocs_far_x4);

		vec3 dof_far_00 = texture(s_FarDoF4, tex_coord_00).xyz;
		vec3 dof_far_10 = texture(s_FarDoF4, tex_coord_10).xyz;
		vec3 dof_far_01 = texture(s_FarDoF4, tex_coord_01).xyz;
		vec3 dof_far_11 = texture(s_FarDoF4, tex_coord_11).xyz;

		vec2 image_coord = FS_IN_TexCoord / u_PixelSize;
		vec2 fractional = fract(image_coord);
		float a = (1.0 - fractional.x) * (1.0 - fractional.y);
		float b = fractional.x * (1.0 - fractional.y);
		float c = (1.0 - fractional.x) * fractional.y;
		float d = fractional.x * fractional.y;

		vec3 dof_far = vec3(0.0);
		float weights_sum = 0.0;

		float weight_00 = a / (cocs_far_diffs.x + 0.001);
		dof_far += weight_00 * dof_far_00;
		weights_sum += weight_00;

		float weight_10 = b / (cocs_far_diffs.y + 0.001);
		dof_far += weight_10 * dof_far_10;
		weights_sum += weight_10;

		float weight_01 = c / (cocs_far_diffs.z + 0.001);
		dof_far += weight_01 * dof_far_01;
		weights_sum += weight_01;

		float weight_11 = d / (cocs_far_diffs.w + 0.001);
		dof_far += weight_11 * dof_far_11;
		weights_sum += weight_11;

		dof_far /= weights_sum;

		result = mix(result, dof_far, u_Blend * coc_far);
	}

	// Near Field
	{
		float coc_near = texture(s_CoCBlur4, FS_IN_TexCoord).x;
		vec3 dof_near = texture(s_NearDoF4, FS_IN_TexCoord).xyz;

		result = mix(result, dof_near, u_Blend * coc_near);
	}

	FS_OUT_FragColor = result;
}

// ----------------------------------------------------------------------