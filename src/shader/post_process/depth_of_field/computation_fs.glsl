layout (location = 0) out vec3 PS_OUT_NearDoF4;
layout (location = 1) out vec3 PS_OUT_FarDoF4;

in vec2 PS_IN_TexCoord;

uniform vec2 u_PixelSize;
uniform float u_KernelSize; 

uniform sampler2D s_CoC4;
uniform sampler2D s_NearBlurCoC4;
uniform sampler2D s_Color4;
uniform sampler2D s_ColorFarCoC4;

const vec2 offsets[] =
{
	2.0f * vec2(1.000000f, 0.000000f),
	2.0f * vec2(0.707107f, 0.707107f),
	2.0f * vec2(-0.000000f, 1.000000f),
	2.0f * vec2(-0.707107f, 0.707107f),
	2.0f * vec2(-1.000000f, -0.000000f),
	2.0f * vec2(-0.707106f, -0.707107f),
	2.0f * vec2(0.000000f, -1.000000f),
	2.0f * vec2(0.707107f, -0.707107f),
	
	4.0f * vec2(1.000000f, 0.000000f),
	4.0f * vec2(0.923880f, 0.382683f),
	4.0f * vec2(0.707107f, 0.707107f),
	4.0f * vec2(0.382683f, 0.923880f),
	4.0f * vec2(-0.000000f, 1.000000f),
	4.0f * vec2(-0.382684f, 0.923879f),
	4.0f * vec2(-0.707107f, 0.707107f),
	4.0f * vec2(-0.923880f, 0.382683f),
	4.0f * vec2(-1.000000f, -0.000000f),
	4.0f * vec2(-0.923879f, -0.382684f),
	4.0f * vec2(-0.707106f, -0.707107f),
	4.0f * vec2(-0.382683f, -0.923880f),
	4.0f * vec2(0.000000f, -1.000000f),
	4.0f * vec2(0.382684f, -0.923879f),
	4.0f * vec2(0.707107f, -0.707107f),
	4.0f * vec2(0.923880f, -0.382683f),

	6.0f * vec2(1.000000f, 0.000000f),
	6.0f * vec2(0.965926f, 0.258819f),
	6.0f * vec2(0.866025f, 0.500000f),
	6.0f * vec2(0.707107f, 0.707107f),
	6.0f * vec2(0.500000f, 0.866026f),
	6.0f * vec2(0.258819f, 0.965926f),
	6.0f * vec2(-0.000000f, 1.000000f),
	6.0f * vec2(-0.258819f, 0.965926f),
	6.0f * vec2(-0.500000f, 0.866025f),
	6.0f * vec2(-0.707107f, 0.707107f),
	6.0f * vec2(-0.866026f, 0.500000f),
	6.0f * vec2(-0.965926f, 0.258819f),
	6.0f * vec2(-1.000000f, -0.000000f),
	6.0f * vec2(-0.965926f, -0.258820f),
	6.0f * vec2(-0.866025f, -0.500000f),
	6.0f * vec2(-0.707106f, -0.707107f),
	6.0f * vec2(-0.499999f, -0.866026f),
	6.0f * vec2(-0.258819f, -0.965926f),
	6.0f * vec2(0.000000f, -1.000000f),
	6.0f * vec2(0.258819f, -0.965926f),
	6.0f * vec2(0.500000f, -0.866025f),
	6.0f * vec2(0.707107f, -0.707107f),
	6.0f * vec2(0.866026f, -0.499999f),
	6.0f * vec2(0.965926f, -0.258818f),
};

// ----------------------------------------------------------------------

vec3 near_field(vec2 tex_coord)
{
	vec3 result = texture(s_Color4, tex_coord).xyz;
	
	for (int i = 0; i < 48; i++)
	{
		vec2 offset = u_KernelSize * offsets[i] * u_PixelSize;
		result += texture(s_Color4, tex_coord + offset).xyz;
	}

	return result / 49.0;
}

// ----------------------------------------------------------------------

vec3 far_field(vec2 tex_coord)
{
	vec3 result = texture(s_ColorFarCoC4, tex_coord).xyz;
	float weights_sum = texture(s_CoC4, tex_coord).y;
	
	for (int i = 0; i < 48; i++)
	{
		vec2 offset = kernelScale * offsets[i] * pixelSize;
		
		float coc_sample = texture(s_Color4, tex_coord + offset).y;
		vec3 sample = texture(s_ColorFarCoC4, tex_coord + offset).xyz;
		
		result += sample; // the texture is pre-multiplied so don't need to multiply here by weight
		weights_sum += coc_sample;
	}

	return result / weights_sum;	
}

// ----------------------------------------------------------------------
// Generate bokeh shapes. 
// ----------------------------------------------------------------------

void main()
{
    float coc_near_blurred = texture(s_NearBlurCoC4, PS_IN_TexCoord).x;
	float coc_far = texture(s_CoC4, PS_IN_TexCoord).y;
	vec3 color = texture(s_Color4, PS_IN_TexCoord).xyz;

	if (coc_near_blurred > 0.0)
		PS_OUT_NearDoF4 = near_field(PS_IN_TexCoord);
	else
		PS_OUT_NearDoF4 = color;

	if (coc_far > 0.0)
		PS_OUT_FarDoF4 = far_field(PS_IN_TexCoord);
	else
		PS_OUT_FarDoF4 = 0.0;
}

// ----------------------------------------------------------------------