#include <../../common/uniforms.glsl>

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
uniform sampler2D s_Velocity;

uniform int u_NumSamples;
uniform float u_Scale;

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
	vec3 result = texture(s_Color, FS_IN_TexCoord).rgb;

	vec2 texel_size = 1.0 / vec2(textureSize(s_Color, 0));
	vec2 velocity = texture(s_Velocity, FS_IN_TexCoord).rg;

	// Remap to [-1, 1] range
	//velocity = velocity * 2.0 - 1.0;
		 
	velocity *= u_Scale;

	float speed = length(velocity / texel_size);
	int num_samples = u_NumSamples;//clamp(int(speed), 1, max_motion_blur_samples);

	for (int i = 0; i < num_samples; i++)
	{
		vec2 offset = velocity * (float(i) / float(num_samples - 1) - 0.5);
		result += texture(s_Color, FS_IN_TexCoord + offset).rgb;
	}

	result =  result / float(num_samples);

	FS_OUT_FragColor = result;
}

// ------------------------------------------------------------------