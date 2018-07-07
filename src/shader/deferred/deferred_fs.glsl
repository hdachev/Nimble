#include <../common/uniforms.glsl>
#include <../common/helper.glsl>
#include <../pbr/pbr.glsl>

// ------------------------------------------------------------------
// SAMPLERS  --------------------------------------------------------
// ------------------------------------------------------------------

uniform sampler2D s_GBufferRT0;
uniform sampler2D s_GBufferRT1;
uniform sampler2D s_GBufferRT2;
uniform sampler2D s_GBufferRT3;
uniform sampler2D s_GBufferRTDepth;
uniform sampler2DArray s_ShadowMap;
uniform samplerCube s_IrradianceMap;
uniform samplerCube s_PrefilteredMap;
uniform sampler2D s_BRDF;
uniform sampler2D s_SSAO;
uniform sampler2D s_SSR;

#include <../csm/csm.glsl>

// ------------------------------------------------------------------
// INPUT VARIABLES  -------------------------------------------------
// ------------------------------------------------------------------

in vec2 PS_IN_TexCoord;
in vec2 PS_IN_ViewRay;

// ------------------------------------------------------------------
// OUTPUT VARIABLES  ------------------------------------------------
// ------------------------------------------------------------------

layout (location = 0) out vec3 PS_OUT_Color;

// ------------------------------------------------------------------
// HELPER FUNCTIONS -------------------------------------------------
// ------------------------------------------------------------------

vec3 get_normal()
{
	// vec2 encoded_normal = texture(s_GBufferRT1, PS_IN_TexCoord).rg;
    // return decode_normal(encoded_normal);

    return normalize(texture(s_GBufferRT2, PS_IN_TexCoord).rgb);
}

// ------------------------------------------------------------------

vec4 get_albedo()
{
    return texture(s_GBufferRT0, PS_IN_TexCoord);
}

// ------------------------------------------------------------------

float get_metalness()
{
    return texture(s_GBufferRT3, PS_IN_TexCoord).x;
}

// ------------------------------------------------------------------

float get_roughness()
{
    return texture(s_GBufferRT3, PS_IN_TexCoord).y;
}

// ------------------------------------------------------------------

vec3 get_position(float ndc_depth)
{
	// float depth = texture(s_GBufferRTDepth, PS_IN_TexCoord).x;
    // float view_z = projMat[3][2] / (2 * depth - 1 - projMat[2][2]);
    
	// float view_x = PS_IN_ViewRay.x * view_z;
    // float view_y = PS_IN_ViewRay.y * view_z;

	// return vec3(view_x, view_y, view_z);
	// Remap depth to [-1.0, 1.0] range. 
	float depth = ndc_depth * 2.0 - 1.0;

	// Take texture coordinate and remap to [-1.0, 1.0] range. 
	vec2 screen_pos = PS_IN_TexCoord * 2.0 - 1.0;

	// // Create NDC position.
	vec4 ndc_pos = vec4(screen_pos, depth, 1.0);

	// Transform back into world position.
	vec4 world_pos = invViewProj * ndc_pos;

	// Undo projection.
	world_pos = world_pos / world_pos.w;

	return world_pos.xyz;
	
	//return texture(s_GBufferRT3, PS_IN_TexCoord).xyz;
}

// ------------------------------------------------------------------

float get_depth()
{
	return texture(s_GBufferRTDepth, PS_IN_TexCoord).x;
}

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
	// Extract values from G-Buffer
	vec4 kAlbedoAlpha = get_albedo();

	if (kAlbedoAlpha.w < 0.5)
		discard;

	vec3 kAlbedo = kAlbedoAlpha.xyz;
	float kMetalness = get_metalness();
	float kRoughness = get_roughness();
	vec3 N = get_normal();
	float frag_depth = get_depth();
	vec3 frag_pos = get_position(frag_depth);
	vec3 view_pos = viewPos.xyz;

	vec3 V = normalize(view_pos - frag_pos); // FragPos -> ViewPos vector
	vec3 R = reflect(-V, N); 
	vec3 Lo = vec3(0.0);

	vec3 F0 = vec3(0.04);
	F0 = mix(F0, kAlbedo, kMetalness);

	float NdotV = max(dot(N, V), 0.0);
	vec3  F = fresnel_schlick_roughness(NdotV, F0, kRoughness);

	vec3 shadow_debug = vec3(0.0);
	float ao = texture(s_SSAO, PS_IN_TexCoord).r;

	// For each directional light...
	{
		vec3 L = normalize(-directionalLight.direction.xyz); // FragPos -> LightPos vector
		vec3 H = normalize(V + L);
		float HdotV = clamp(dot(H, V), 0.0, 1.0);
		float NdotH = max(dot(N, H), 0.0);
		float NdotL = max(dot(N, L), 0.0);

		// Shadows ------------------------------------------------------------------
		float shadow = shadow_occlussion(frag_depth, frag_pos, N, L);

		shadow_debug = debug_color(frag_depth);

		// Radiance -----------------------------------------------------------------

		vec3 Li = directionalLight.color.xyz * directionalLight.color.w;

		// --------------------------------------------------------------------------

		// Specular Term ------------------------------------------------------------
		float D = distribution_trowbridge_reitz_ggx(NdotH, kRoughness);
		float G = geometry_smith(NdotV, NdotL, kRoughness);

		vec3 numerator = D * G * F;
		float denominator = 4.0 * NdotV * NdotL; 

		vec3 specular = numerator / max(denominator, 0.001);
		// --------------------------------------------------------------------------

		// Diffuse Term -------------------------------------------------------------
		vec3 diffuse = kAlbedo / kPI;
		// --------------------------------------------------------------------------

		// Combination --------------------------------------------------------------
		vec3 kS = F;
		vec3 kD = vec3(1.0) - kS;
		kD *= 1.0 - kMetalness;

		Lo += shadow * (kD * kAlbedo / kPI + specular) * Li * NdotL;
		// --------------------------------------------------------------------------
	}

	// For each point light...
	for (int i = 0; i < pointLightCount; i++)
	{
		vec3 L = normalize(pointLights[i].position.xyz - frag_pos); // FragPos -> LightPos vector
		vec3 H = normalize(V + L);
		float HdotV = clamp(dot(H, V), 0.0, 1.0);
		float NdotH = max(dot(N, H), 0.0);
		float NdotL = max(dot(N, L), 0.0);

		// Radiance -----------------------------------------------------------------

		float distance = length(pointLights[i].position.xyz - frag_pos);
		float attenuation = 1.0 / (distance * distance);
		vec3 Li = pointLights[i].color.xyz * attenuation;

		// --------------------------------------------------------------------------

		// Specular Term ------------------------------------------------------------
		float D = distribution_trowbridge_reitz_ggx(NdotH, kRoughness);
		float G = geometry_smith(NdotV, NdotL, kRoughness);

		vec3 numerator = D * G * F;
		float denominator = 4.0 * NdotV * NdotL; 

		vec3 specular = numerator / max(denominator, 0.001);
		// --------------------------------------------------------------------------

		// Diffuse Term -------------------------------------------------------------
		vec3 diffuse = kAlbedo / kPI;
		// --------------------------------------------------------------------------

		// Combination --------------------------------------------------------------
		vec3 kS = F;
		vec3 kD = vec3(1.0) - kS;
		kD *= 1.0 - kMetalness;

		Lo += (kD * kAlbedo / kPI + specular) * Li * NdotL;
		// --------------------------------------------------------------------------
	}

	vec3 kS = F;
	vec3 kD = 1.0 - kS;
	kD *= 1.0 - kMetalness;

	vec3 irradiance = texture(s_IrradianceMap, N).rgb;
	vec3 diffuse = irradiance * kAlbedo;

	// Sample prefilter map and BRDF LUT
	vec3 ssr = texture(s_SSR, PS_IN_TexCoord).rgb;
	vec3 prefilteredColor = vec3(0.0);

	if (ssr.x > 0.0 || ssr.y > 0.0 || ssr.z > 0.0)
		prefilteredColor = ssr;
	else
		prefilteredColor = textureLod(s_PrefilteredMap, R, kRoughness * kMaxLOD).rgb;

	vec2 brdf = texture(s_BRDF, vec2(max(NdotV, 0.0), kRoughness)).rg;
	vec3 specular = prefilteredColor * (F * brdf.x + brdf.y);

	vec3 ambient = (kD * diffuse + specular) * kAmbient * ao;
	vec3 color = Lo + ambient;
	
    PS_OUT_Color = color;
}