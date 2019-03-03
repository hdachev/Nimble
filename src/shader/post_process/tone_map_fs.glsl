#include <../common/uniforms.glsl>
#include <../common/helper.glsl>

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
// CONSTANTS --------------------------------------------------------
// ------------------------------------------------------------------

// Global constants for Uncharted 2 tone mapping.
const float A = 0.15;
const float B = 0.50;
const float C = 0.10;
const float D = 0.20;
const float E = 0.02;
const float F = 0.30;
const float W = 11.2;

// ------------------------------------------------------------------
// FUNCTIONS --------------------------------------------------------
// ------------------------------------------------------------------

vec3 linear_tone_mapping(vec3 color, float exposure)
{
	return color * exposure;
}

// ------------------------------------------------------------------

vec3 reinhard_tone_mapping(vec3 color, float exposure)
{
	vec3 ret_color = color * exposure;
	ret_color = ret_color / (1.0 + ret_color);

	return ret_color;
}

// ------------------------------------------------------------------

vec3 haarm_peter_duiker_tone_mapping(vec3 color, float exposure, sampler2D lut)
{
	vec3 exp_color = color * exposure;

	vec3 ld = vec3(0.002);
	float lin_reference = 0.18;
	float log_reference = 444;
	float log_gamma = 0.45;

	vec3 log_color = (log10(0.4 * exp_color / lin_reference) / ld * log_gamma + log_reference) / 1023.0;
	log_color = clamp(log_color, 0.0, 1.0);

	float film_lut_width = 256;
   	float padding = 0.5 / film_lut_width;
      
	//  apply response lookup and color grading for target display
	vec3 ret_color;
	ret_color.r = texture(lut, vec2(mix(padding, 1.0 - padding, log_color.r), 0.5)).r;
	ret_color.g = texture(lut, vec2(mix(padding, 1.0 - padding, log_color.g), 0.5)).r;
	ret_color.b = texture(lut, vec2(mix(padding, 1.0 - padding, log_color.b), 0.5)).r;

	return ret_color;
}

// ------------------------------------------------------------------

vec3 filmic_tone_mapping(vec3 color, float exposure)
{
	vec3 exp_color = color * exposure;
	vec3 x = max(vec3(0.0), exp_color - vec3(0.004));
   	vec3 ret_color = (x * (6.2 * x + 0.5)) / (x * (6.2 * x + 1.7) + 0.06);

	return ret_color; 
}

// ------------------------------------------------------------------

vec3 uncharted_2_tone_mapping(vec3 x)
{
	return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}

// ------------------------------------------------------------------

vec3 uncharted_2_tone_mapping(vec3 color, float exposure, float exposure_bias)
{
	vec3 tex_color = color * exposure;
	vec3 curr = uncharted_2_tone_mapping(exposure_bias * tex_color);

   	vec3 white_scale = 1.0f/uncharted_2_tone_mapping(vec3(W));
   	vec3 ret_color = curr * white_scale;

	return ret_color;
}

// ------------------------------------------------------------------

vec3 gamma_correction(vec3 color)
{
	return pow(color, vec3(1.0/2.2));
}

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------


void main(void)
{
    // Get HDR color
	vec3 linear_color = texture(s_Texture, FS_IN_TexCoord).rgb;
	vec3 tone_mapped_color = vec3(1.0);

    float exposure = 1.0;
    // tone_mapped_color = reinhard_tone_mapping(linear_color, exposure);
	tone_mapped_color = uncharted_2_tone_mapping(linear_color, exposure, 2.0);

    FS_OUT_Color = vec4(gamma_correction(tone_mapped_color), 1.0);
}

// ------------------------------------------------------------------