#include <../../common/uniforms.glsl>

out vec4 FragColor;

in vec2 PS_IN_TexCoord;

#define SHOW_FORWARD_COLOR 0
#define SHOW_FORWARD_DEPTH 1
#define SHOW_DEFERRED_COLOR 2
#define SHOW_GBUFFER_ALBEDO 3
#define SHOW_GBUFFER_NORMALS 4
#define SHOW_GBUFFER_ROUGHNESS 5
#define SHOW_GBUFFER_METALNESS 6
#define SHOW_GBUFFER_VELOCITY 7
#define SHOW_GBUFFER_EMISSIVE 8
#define SHOW_GBUFFER_DISPLACEMENT 9
#define SHOW_GBUFFER_DEPTH 10
#define SHOW_SHADOW_MAPS 11
#define SHOW_SSAO 20
#define SHOW_SSAO_BLUR 21
#define SHOW_BRIGHT_PASS 22
#define SHOW_SSR 23

uniform float u_FarPlane;
uniform float u_NearPlane;

uniform sampler2D s_Color;
uniform sampler2D s_Depth;
uniform sampler2DArray s_CSMShadowMaps;
uniform sampler2D s_GBufferRT0;
uniform sampler2D s_GBufferRT1;
uniform sampler2D s_GBufferRT2;
uniform sampler2D s_GBufferRT3;
uniform sampler2D s_GBufferRTDepth;
uniform sampler2D s_DeferredColor;
uniform sampler2D s_SSAO;
uniform sampler2D s_SSAO_Blur;
uniform sampler2D s_BrightPass;
uniform sampler2D s_SSR;

float get_linear_depth(sampler2D depth_sampler)
{
    float f = u_FarPlane;
    float n = u_NearPlane;
    float z = (2 * n) / (f + n - texture( depth_sampler, PS_IN_TexCoord ).x * (f - n));
    return z;
}

vec4 visualize_forward_depth()
{
	float depth = get_linear_depth(s_Depth);
	return vec4(vec3(depth), 1.0);
}

vec3 decode_normal(vec2 enc)
{
    vec2 fenc = enc * 4.0 - 2.0;
    float f = dot(fenc, fenc);
    float g = sqrt(1.0 - f / 4.0);
    vec3 n;
    n.xy = fenc * g;
    n.z = 1 - f / 2.0;
    return n;
}

vec4 visualize_gbuffer_albedo()
{
	return texture(s_GBufferRT0, PS_IN_TexCoord);
} 

vec4 visualize_gbuffer_normals()
{
	// vec2 encoded_normal = texture(s_GBufferRT1, PS_IN_TexCoord).xy;
	// vec3 n = decode_normal(encoded_normal);
	
	// // Remap to 0 - 1 range.
	// n = (n + vec3(1.0)) / 2.0;

	vec3 n = normalize(texture(s_GBufferRT2, PS_IN_TexCoord).xyz);
	
	// Remap to 0 - 1 range.
	n = (n + vec3(1.0)) / 2.0;

	return vec4(n, 1.0);
}

vec4 visualize_gbuffer_metalness()
{
	float metalness = texture(s_GBufferRT3, PS_IN_TexCoord).x;
	return vec4(vec3(metalness), 1.0);
} 

vec4 visualize_gbuffer_roughness()
{
	float roughness = texture(s_GBufferRT3, PS_IN_TexCoord).y;
	return vec4(vec3(roughness), 1.0);
}

vec4 visualize_gbuffer_velocity()
{
	vec2 velocity = texture(s_GBufferRT1, PS_IN_TexCoord).zw;

	//velocity = pow(velocity, 1.0/3.0);
	velocity = velocity * 2.0 - 1.0;

	return vec4(velocity, 0.0, 1.0);
}

vec4 visualize_gbuffer_depth()
{
	float depth = get_linear_depth(s_GBufferRTDepth);
	return vec4(vec3(depth), 1.0);
}

vec4 visualize_ssao()
{
	return vec4(vec3(texture(s_SSAO, PS_IN_TexCoord).r), 1.0);
}

vec4 visualize_ssao_blur()
{
	return vec4(vec3(texture(s_SSAO_Blur, PS_IN_TexCoord).r), 1.0);
}

vec4 visualize_bright_pass()
{
	return vec4(texture(s_BrightPass, PS_IN_TexCoord).rgb, 1.0);
}

vec4 visualize_ssr()
{
	return vec4(texture(s_SSR, PS_IN_TexCoord).rgb, 1.0);
}

void main()
{
	if (current_output == SHOW_FORWARD_COLOR)
		FragColor = vec4(texture(s_Color, PS_IN_TexCoord).xyz, 1.0);
	else if (current_output == SHOW_FORWARD_DEPTH)
		FragColor = visualize_forward_depth();
	else if (current_output == SHOW_DEFERRED_COLOR)
		FragColor = vec4(texture(s_DeferredColor, PS_IN_TexCoord).xyz, 1.0);
	else if (current_output == SHOW_GBUFFER_ALBEDO)
		FragColor = visualize_gbuffer_albedo();
	else if (current_output == SHOW_GBUFFER_NORMALS)
		FragColor = visualize_gbuffer_normals();
	else if (current_output == SHOW_GBUFFER_ROUGHNESS)
		FragColor = visualize_gbuffer_roughness();
	else if (current_output == SHOW_GBUFFER_METALNESS)
		FragColor = visualize_gbuffer_metalness();
	else if (current_output == SHOW_GBUFFER_VELOCITY)
		FragColor = visualize_gbuffer_velocity();
	else if (current_output == SHOW_GBUFFER_DEPTH)
		FragColor = visualize_gbuffer_depth();
	else if (current_output == SHOW_SSAO)
		FragColor = visualize_ssao();
	else if (current_output == SHOW_SSAO_BLUR)
		FragColor = visualize_ssao_blur();
	else if (current_output == SHOW_BRIGHT_PASS)
		FragColor = visualize_bright_pass();
	else if (current_output == SHOW_SSR)
		FragColor = visualize_ssr();
	else if (current_output >= SHOW_SHADOW_MAPS)
		FragColor = vec4(vec3(texture(s_CSMShadowMaps, vec3(PS_IN_TexCoord, float(current_output - SHOW_SHADOW_MAPS))).x), 1.0);
	else 
		FragColor = vec4(1.0);
}