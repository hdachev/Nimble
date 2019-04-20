// ------------------------------------------------------------------
// INPUT VARIABLES  -------------------------------------------------
// ------------------------------------------------------------------

in vec2 FS_IN_TexCoord;

// ------------------------------------------------------------------
// OUTPUT VARIABLES  ------------------------------------------------
// ------------------------------------------------------------------

layout (location = 0) out vec3 FS_OUT_Color;
layout (location = 1) out vec3 FS_OUT_MulCoCFar;
layout (location = 2) out vec2 FS_OUT_CoC;

// ------------------------------------------------------------------
// UNIFORMS  --------------------------------------------------------
// ------------------------------------------------------------------

uniform vec2 u_PixelSize;

uniform sampler2D s_Color;
uniform sampler2D s_CoC;

// ----------------------------------------------------------------------
// Downsamples Color and Near/Far CoC along with CoC Far weighted Color.
// ----------------------------------------------------------------------

void main()
{
	vec2 tex_coord_00 = FS_IN_TexCoord + vec2(-0.25f, -0.25f) * u_PixelSize;
	vec2 tex_coord_10 = FS_IN_TexCoord + vec2( 0.25f, -0.25f) * u_PixelSize;
	vec2 tex_coord_01 = FS_IN_TexCoord + vec2(-0.25f,  0.25f) * u_PixelSize;
	vec2 tex_coord_11 = FS_IN_TexCoord + vec2( 0.25f,  0.25f) * u_PixelSize;

	vec3 color = texture(s_Color, FS_IN_TexCoord).xyz;
	vec2 coc = texture(s_CoC, tex_coord_00).xy;
	
	// custom bilinear filtering of color weighted by coc far
	
	float coc_far_00 = texture(s_CoC, tex_coord_00).y;
	float coc_far_10 = texture(s_CoC, tex_coord_10).y;
	float coc_far_01 = texture(s_CoC, tex_coord_01).y;
	float coc_far_11 = texture(s_CoC, tex_coord_11).y;

	float weight_00 = 1000.0;
	vec3 color_mul_coc_far = weight_00 * texture(s_Color, tex_coord_00).xyz;
	float weights_sum = weight_00;
	
	float weight_10 = 1.0 / (abs(coc_far_00 - coc_far_10) + 0.001);
	color_mul_coc_far += weight_10 * texture(s_Color, tex_coord_10).xyz;
	weights_sum += weight_10;
	
	float weight_01 = 1.0 / (abs(coc_far_00 - coc_far_01) + 0.001);
	color_mul_coc_far += weight_01 * texture(s_Color, tex_coord_01).xyz;
	weights_sum += weight_01;
	
	float weight_11 = 1.0 / (abs(coc_far_00 - coc_far_11) + 0.001);
	color_mul_coc_far += weight_11 * texture(s_Color, tex_coord_11).xyz;
	weights_sum += weight_11;

	color_mul_coc_far /= weights_sum;
	color_mul_coc_far *= coc.y;

	FS_OUT_Color = color;
	FS_OUT_MulCoCFar = color_mul_coc_far;
	FS_OUT_CoC = coc;
}

// ----------------------------------------------------------------------