#include <../common/uniforms.glsl>

// ------------------------------------------------------------------
// SAMPLERS  --------------------------------------------------------
// ------------------------------------------------------------------

#ifdef ALBEDO_TEXTURE
    uniform sampler2D s_Albedo;
#endif

#ifdef NORMAL_TEXTURE
    uniform sampler2D s_Normal;
#endif

#ifdef METALNESS_TEXTURE
    uniform sampler2D s_Metalness;
#endif

#ifdef ROUGHNESS_TEXTURE
    uniform sampler2D s_Roughness;
#endif

#ifdef HEIGHT_TEXTURE
    uniform sampler2D s_Displacement;
#endif

#ifdef EMISSIVE_TEXTURE
    uniform sampler2D s_Emissive;
#endif

uniform samplerCube s_IrradianceMap;
uniform samplerCube s_PrefilteredMap;
uniform sampler2D s_BRDF;
uniform sampler2DArray s_ShadowMap;

// ------------------------------------------------------------------
// INPUT VARIABLES  -------------------------------------------------
// ------------------------------------------------------------------

in vec3 PS_IN_Position;
in vec4 PS_IN_NDCFragPos;
in vec3 PS_IN_CamPos;
in vec3 PS_IN_Normal;
in vec2 PS_IN_TexCoord;

#ifdef NORMAL_TEXTURE
	in vec3 PS_IN_Tangent;
	in vec3 PS_IN_Bitangent;
#endif

#ifdef HEIGHT_TEXTURE
	in vec3 PS_IN_TangentViewPos;
	in vec3 PS_IN_TangentFragPos;
#endif

#include <../csm/csm.glsl>
#include <../common/helper.glsl>
#include <../pbr/pbr.glsl>

// ------------------------------------------------------------------
// OUTPUT VARIABLES  ------------------------------------------------
// ------------------------------------------------------------------

layout (location = 0) out vec4 PS_OUT_Color;

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
#ifdef HEIGHT_TEXTURE
	vec2 tex_coord = parallax_occlusion_tex_coords(normalize(PS_IN_TangentViewPos), normalize(PS_IN_TangentFragPos), PS_IN_TexCoord, 0.1, s_Displacement); 
#else
	vec2 tex_coord = PS_IN_TexCoord;
#endif

	vec4 kAlbedoAlpha = texture(s_Albedo, tex_coord);
	float kMetalness = texture(s_Metalness, tex_coord).x; 
	float kRoughness = texture(s_Roughness, tex_coord).x; 

	if (kAlbedoAlpha.w < 0.5)
		discard;

	vec3 kAlbedo = kAlbedoAlpha.xyz;

#ifdef NORMAL_TEXTURE
	vec3 N = get_normal_from_map(PS_IN_Tangent, PS_IN_Bitangent, PS_IN_Normal, tex_coord, s_Normal);
#else
	vec3 N = PS_IN_Normal;
#endif

	vec3 V = normalize(PS_IN_CamPos - PS_IN_Position); // FragPos -> ViewPos vector
	vec3 R = reflect(-V, N); 
	vec3 Lo = vec3(0.0);

	vec3 F0 = vec3(0.04);
	F0 = mix(F0, kAlbedo, kMetalness);

	float NdotV = max(dot(N, V), 0.0);
	vec3  F = fresnel_schlick_roughness(NdotV, F0, kRoughness);

	vec3 shadow_debug = vec3(0.0);

	// For each directional light...
	{
		vec3 L = normalize(-directionalLight.direction.xyz); // FragPos -> LightPos vector
		vec3 H = normalize(V + L);
		float HdotV = clamp(dot(H, V), 0.0, 1.0);
		float NdotH = max(dot(N, H), 0.0);
		float NdotL = max(dot(N, L), 0.0);

		// Shadows ------------------------------------------------------------------
		float frag_depth = (PS_IN_NDCFragPos.z / PS_IN_NDCFragPos.w) * 0.5 + 0.5;
		float shadow = shadow_occlussion(frag_depth, PS_IN_Position, PS_IN_Normal, L);

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
		vec3 L = normalize(pointLights[i].position.xyz - PS_IN_Position); // FragPos -> LightPos vector
		vec3 H = normalize(V + L);
		float HdotV = clamp(dot(H, V), 0.0, 1.0);
		float NdotH = max(dot(N, H), 0.0);
		float NdotL = max(dot(N, L), 0.0);

		// Radiance -----------------------------------------------------------------

		float distance = length(pointLights[i].position.xyz - PS_IN_Position);
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
	vec3 prefilteredColor = textureLod(s_PrefilteredMap, R, kRoughness * kMaxLOD).rgb;
	vec2 brdf = texture(s_BRDF, vec2(max(NdotV, 0.0), kRoughness)).rg;
	vec3 specular = prefilteredColor * (F * brdf.x + brdf.y);

	vec3 ambient = (kD * diffuse + specular) * kAmbient;
	// vec3 ambient = vec3(0.03) * diffuse * kAmbient;

	vec3 color = Lo + ambient;

	// Gamma Correction
	color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0/2.2));  

    PS_OUT_Color = vec4(color, 1.0);
}

// ------------------------------------------------------------------