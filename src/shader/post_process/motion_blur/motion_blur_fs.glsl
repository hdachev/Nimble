#include <../../common/uniforms.glsl>

out vec3 FragColor;

in vec2 PS_IN_TexCoord;

uniform sampler2D s_ColorMap;
uniform sampler2D s_VelocityMap;

void main()
{
	vec3 result = texture(s_ColorMap, PS_IN_TexCoord).rgb;

	if (motion_blur == 1)
	{
		vec2 texel_size = 1.0 / vec2(textureSize(s_ColorMap, 0));
		vec2 velocity = vec2(0.0);

		if (renderer == 0)
			velocity = texture(s_VelocityMap, PS_IN_TexCoord).rg;
		else if (renderer == 1)
		 	velocity = texture(s_VelocityMap, PS_IN_TexCoord).ba;

		// Remap to [-1, 1] range
		//velocity = velocity * 2.0 - 1.0;
			 
		velocity *= velocity_scale;

		float speed = length(velocity / texel_size);
		int num_samples = max_motion_blur_samples;//clamp(int(speed), 1, max_motion_blur_samples);

		for (int i = 0; i < num_samples; i++)
		{
			vec2 offset = velocity * (float(i) / float(num_samples - 1) - 0.5);
			result += texture(s_ColorMap, PS_IN_TexCoord + offset).rgb;
		}

		result =  result / float(num_samples);
	}

	FragColor = result;
}