#include <../common/uniforms.glsl>

// ------------------------------------------------------------------
// SAMPLERS ---------------------------------------------------------
// ------------------------------------------------------------------

uniform samplerCube s_IrradianceMap;
uniform samplerCube s_PrefilteredMap;
uniform sampler2D s_BRDF;
uniform sampler2DArray s_ShadowMap;

// ------------------------------------------------------------------
// INPUT VARIABLES  -------------------------------------------------
// ------------------------------------------------------------------

in vec3 PS_IN_Position;
in vec4 PS_IN_NDCFragPos;
in vec4 PS_IN_ScreenPosition;
in vec4 PS_IN_LastScreenPosition;
in vec3 PS_IN_Normal;
in vec2 PS_IN_TexCoord;

#ifdef TEXTURE_NORMAL
	in vec3 PS_IN_Tangent;
	in vec3 PS_IN_Bitangent;
#endif

#ifdef DISPLACEMENT_TYPE_PARALLAX_OCCLUSION
	in vec3 PS_IN_TangentViewPos;
	in vec3 PS_IN_TangentFragPos;
#endif

#include <../common/helper.glsl>
#include <../common/material.glsl>
#include <../shadows/pcf.glsl>
#include <../pbr/pbr.glsl>

// ------------------------------------------------------------------
// OUTPUT VARIABLES  ------------------------------------------------
// ------------------------------------------------------------------

layout (location = 0) out vec3 PS_OUT_Color;
// layout (location = 1) out vec2 PS_OUT_Velocity;

// ------------------------------------------------------------------
// FUNCTIONS --------------------------------------------------------
// ------------------------------------------------------------------

void fill_fragment_properties(inout FragmentProperties f)
{
	f.Position = PS_IN_Position;
	f.Normal = PS_IN_Normal;
	f.FragDepth = (PS_IN_NDCFragPos.z / PS_IN_NDCFragPos.w) * 0.5 + 0.5;
#ifdef TEXTURE_NORMAL
	f.Tangent = PS_IN_Tangent;
	f.Bitangent = PS_IN_Bitangent;
#endif
#ifdef DISPLACEMENT_TYPE_PARALLAX_OCCLUSION
	f.TangentViewPos = PS_IN_TangentViewPos;
	f.TangentFragPos = PS_IN_TangentFragPos;
	f.TexCoords = get_parallax_occlusion_texcoords(PS_IN_TexCoord, PS_IN_TangentViewPos, PS_IN_TangentFragPos);
#else
	f.TexCoords = PS_IN_TexCoord;
#endif
}

// ------------------------------------------------------------------

#ifndef FRAGMENT_SHADER_FUNC
#define FRAGMENT_SHADER_FUNC

void fragment_func(inout MaterialProperties m, inout FragmentProperties f)
{
	m.albedo = get_albedo(f.TexCoords);
	m.normal = get_normal(f);
	m.metallic = get_metallic(f.TexCoords);
	m.roughness = get_roughness(f.TexCoords);
}

#endif

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
	FragmentProperties f;

	fill_fragment_properties(f);

	MaterialProperties m;

	// Set material properties
	fragment_func(m, f);

#ifdef BLEND_MODE_MASKED
	// Discard fragments below alpha threshold
	if (m.albedo.w < 0.1)
		discard;
#endif

	PBRProperties pbr;

	// Set PBR properties
	pbr.N = m.normal;
	pbr.V = normalize(viewPos.xyz - f.Position); // FragPos -> ViewPos vector
	pbr.R = reflect(-pbr.V, pbr.N); 
	pbr.F0 = vec3(0.04);
	pbr.F0 = mix(pbr.F0, m.albedo.xyz, m.metallic);
	pbr.NdotV = max(dot(pbr.N, pbr.V), 0.0);
	pbr.F = fresnel_schlick_roughness(pbr.NdotV, pbr.F0, m.roughness);
	pbr.kS = pbr.F;
	pbr.kD = vec3(1.0) - pbr.kS;
	pbr.kD *= 1.0 - m.metallic;

	// Output radiance
	vec3 Lo = vec3(0.0);

	// Add directional light contribution
	Lo += pbr_directional_lights(m, f, pbr);

	// Add point light contributions
	Lo += pbr_point_lights(m, f, pbr);

	// Add spot light contributions
	Lo += pbr_spot_lights(m, f, pbr);

	vec3 irradiance = texture(s_IrradianceMap, pbr.N).rgb;
	vec3 diffuse = irradiance * m.albedo.xyz;

	// Sample prefilter map and BRDF LUT
	vec3 prefilteredColor = textureLod(s_PrefilteredMap, pbr.R, m.roughness * kMaxLOD).rgb;
	vec2 brdf = texture(s_BRDF, vec2(max(pbr.NdotV, 0.0), m.roughness)).rg;
	vec3 specular = prefilteredColor * (pbr.F * brdf.x + brdf.y);

	vec3 ambient = (pbr.kD * diffuse + specular) * kAmbient;
	vec3 color = Lo + ambient;

	PS_OUT_Color = m.albedo.xyz;
    // PS_OUT_Color = color;
	// PS_OUT_Velocity = motion_vector(PS_IN_LastScreenPosition, PS_IN_ScreenPosition);
}

// ------------------------------------------------------------------