out vec4 PS_OUT_FragColor;

in vec2 PS_IN_TexCoord;

uniform vec2 u_PixelSize;
uniform int u_From;
uniform int u_To;

uniform sampler2D s_Texture;

// ----------------------------------------------------------------------
// A general filter that provides copy, min/max downsampling, blur. 
// ----------------------------------------------------------------------

void main()
{
#ifdef COPY
	return texture(s_Texture, PS_IN_TexCoord);
#endif
	
#if CHANNELS_COUNT == 1
	#define CHANNELS x
	float result = 0.0f;
#elif CHANNELS_COUNT == 2
	#define CHANNELS xy
	vec2 result = 0.0f;
#elif CHANNELS_COUNT == 3
	#define CHANNELS xyy
	vec3 result = 0.0f;
#elif CHANNELS_COUNT == 4
	#define CHANNELS xyzw
	vec4 result = 0.0f;
#endif
	
	vec2 direction = 0.0f;
#ifdef HORIZONTAL
	direction = vec2(1.0f, 0.0f);
#endif
#ifdef VERTICAL
	direction = vec2(0.0f, 1.0f);
#endif

#ifdef MIN
	result = texture(s_Texture, PS_IN_TexCoord).CHANNELS;
	for (int i = from; i <= to; i++)
		result = min(result, texture(s_Texture, PS_IN_TexCoord + i * direction * u_PixelSize).CHANNELS);
#endif
#ifdef MIN13
	result = texture(s_Texture, PS_IN_TexCoord).CHANNELS;
	for (int i = -6; i <= 6; i++)
		result = min(result, texture(s_Texture, PS_IN_TexCoord + i * direction * u_PixelSize).CHANNELS);
#endif

#ifdef MAX
	result = texture(s_Texture, PS_IN_TexCoord).CHANNELS;
	for (int i = from; i <= to; i++)
		result = max(result, texture(s_Texture, PS_IN_TexCoord + i * direction * u_PixelSize).CHANNELS);
#endif
#ifdef MAX13
	result = texture(s_Texture, PS_IN_TexCoord).CHANNELS;
	for (int i = -6; i <= 6; i++)
		result = max(result, texture(s_Texture, PS_IN_TexCoord + i * direction * u_PixelSize).CHANNELS);
#endif

#ifdef BLUR
	for (int i = from; i <= to; i++)
		result += texture(s_Texture, PS_IN_TexCoord + i * direction * u_PixelSize).CHANNELS;
	result /= (to - from + 1.0f);
#endif
#ifdef BLUR13
	for (int i = -6; i <= 6; i++)
		result += texture(s_Texture, PS_IN_TexCoord + i * direction * u_PixelSize).CHANNELS;
	result /= 13.0f;
#endif

#if CHANNELS_COUNT == 1
	PS_OUT_FragColor = result.xxxx;
#elif CHANNELS_COUNT == 2
	PS_OUT_FragColor = vec4(result.xy, 0.0f, 0.0f);
#elif CHANNELS_COUNT == 3
	PS_OUT_FragColor = vec4(result.xyz, 0.0f);
#elif CHANNELS_COUNT == 4
	PS_OUT_FragColor = result.xyzw;
#endif
}