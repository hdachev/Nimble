out vec4 PS_OUT_FragColor;

in vec2 PS_IN_TexCoord;

#ifndef COPY
uniform vec2 u_PixelSize;

#ifndef MIN13
#ifndef MAX13
#ifndef BLUR13
uniform int u_From;
uniform int u_To;
#endif
#endif
#endif

#endif

uniform sampler2D s_Texture;

// ----------------------------------------------------------------------
// A general filter that provides copy, min/max downsampling, blur. 
// ----------------------------------------------------------------------

void main()
{
#ifdef COPY
	PS_OUT_FragColor = texture(s_Texture, PS_IN_TexCoord);
#endif
	
#ifdef CHANNELS_COUNT_1
	#define CHANNELS x
	float result = 0.0;
#elif CHANNELS_COUNT_2
	#define CHANNELS xy
	vec2 result = vec2(0.0);
#elif CHANNELS_COUNT_3
	#define CHANNELS xyy
	vec3 result = vec3(0.0);
#elif CHANNELS_COUNT_4
	#define CHANNELS xyzw
	vec4 result = vec4(0.0);
#endif
	
	vec2 direction = vec2(0.0);
#ifdef HORIZONTAL
	direction = vec2(1.0, 0.0);
#endif
#ifdef VERTICAL
	direction = vec2(0.0, 1.0);
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
	result /= (to - from + 1.0);
#endif
#ifdef BLUR13
	for (int i = -6; i <= 6; i++)
		result += texture(s_Texture, PS_IN_TexCoord + i * direction * u_PixelSize).CHANNELS;
	result /= 13.0;
#endif

#ifdef CHANNELS_COUNT_1
	PS_OUT_FragColor = result.xxxx;
#elif CHANNELS_COUNT_2
	PS_OUT_FragColor = vec4(result.xy, 0.0, 0.0);
#elif CHANNELS_COUNT_3
	PS_OUT_FragColor = vec4(result.xyz, 0.0);
#elif CHANNELS_COUNT_4
	PS_OUT_FragColor = result.xyzw;
#endif
}