#include <../../common/uniforms.glsl>
#include <../../common/helper.glsl>

#define BLUR_DEPTH_FACTOR 0.5
#define GAUSS_BLUR_DEVIATION 1.5        
#define HALF_RES_BLUR_KERNEL_SIZE 5
#define DEPTH_LOD 1.0

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

uniform sampler2D s_Depth;
uniform sampler2D s_Volumetric;

uniform vec2 u_Direction;
uniform vec2 u_PixelSize;

// ------------------------------------------------------------------
// FUNCTIONS --------------------------------------------------------
// ------------------------------------------------------------------

const float kPI = 3.14159265359;

float gaussian_weight(float offset, float deviation)
{
	float weight = 1.0 / sqrt(2.0 * kPI * deviation * deviation);
	weight *= exp(-(offset * offset) / (2.0 * deviation * deviation));
	return weight;
}

// ------------------------------------------------------------------

vec3 bilateral_blur(vec2 tex_coord, vec2 direction, int radius, vec2 pixel_size)
{
	//const float deviation = kernelRadius / 2.5;
	const float deviation = float(radius) / GAUSS_BLUR_DEVIATION; // make it really strong

	vec4 center_color = texture(s_Volumetric, tex_coord);
	vec3 color = center_color.xyz;
	//return float4(color, 1);
	float center_depth = depth_exp_to_view(near_plane, far_plane, textureLod(s_Depth, tex_coord, DEPTH_LOD).r);

	float weight_sum = 0;

	// gaussian weight is computed from constants only -> will be computed in compile time
    float weight = gaussian_weight(0.0, deviation);
	color *= weight;
	weight_sum += weight;

	vec2 offset_direction = direction * pixel_size;
				
	for (int i = -radius; i < 0; i++)
	{
        vec2 uv = tex_coord + offset_direction * float(i);
        vec3 sample_color = texture(s_Volumetric, uv).rgb;
        float sample_depth = depth_exp_to_view(near_plane, far_plane, textureLod(s_Depth, uv, DEPTH_LOD).r);

		float depth_diff = abs(center_depth - sample_depth);
        float dFactor = depth_diff * BLUR_DEPTH_FACTOR;
		float w = exp(-(dFactor * dFactor));

		// gaussian weight is computed from constants only -> will be computed in compile time
		weight = gaussian_weight(float(i), deviation) * w;

		color += weight * sample_color;
		weight_sum += weight;
	}

	for (int i = 1; i <= radius; i++)
	{
		vec2 uv = tex_coord + offset_direction * float(i);
        vec3 sample_color = texture(s_Volumetric, uv).rgb;
        float sample_depth = depth_exp_to_view(near_plane, far_plane, textureLod(s_Depth, uv, DEPTH_LOD).r);

		float depth_diff = abs(center_depth - sample_depth);
        float dFactor = depth_diff * BLUR_DEPTH_FACTOR;
		float w = exp(-(dFactor * dFactor));
		
		// gaussian weight is computed from constants only -> will be computed in compile time
		weight = gaussian_weight(float(i), deviation) * w;

		color += weight * sample_color;
		weight_sum += weight;
	}

	color /= weight_sum;
	return color;
}

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
	FS_OUT_FragColor = bilateral_blur(FS_IN_TexCoord, u_Direction, HALF_RES_BLUR_KERNEL_SIZE, u_PixelSize);
}

// ------------------------------------------------------------------