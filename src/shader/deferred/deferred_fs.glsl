#include <../common/uniforms.glsl>

// ------------------------------------------------------------------
// SAMPLERS ---------------------------------------------------------
// ------------------------------------------------------------------

uniform samplerCube s_IrradianceMap;
uniform samplerCube s_PrefilteredMap;
uniform sampler2D s_BRDF;
uniform sampler2DArray s_ShadowMap;

uniform sampler2D s_GBufferRT1;
uniform sampler2D s_GBufferRT2;
uniform sampler2D s_GBufferRT3;
uniform sampler2D s_GBufferRT4;
uniform sampler2D s_Depth;
uniform sampler2D s_SSAO;

// ------------------------------------------------------------------
// INPUT VARIABLES  -------------------------------------------------
// ------------------------------------------------------------------

in vec2 FS_IN_TexCoord;

#include <../common/helper.glsl>
#include <../common/material.glsl>
#include <../pbr/pbr.glsl>

// ------------------------------------------------------------------
// OUTPUT VARIABLES  ------------------------------------------------
// ------------------------------------------------------------------

layout (location = 0) out vec3 FS_OUT_Color;

// ------------------------------------------------------------------
// FUNCTIONS --------------------------------------------------------
// ------------------------------------------------------------------

vec3 unpack_normal()
{
    return normalize(texture(s_GBufferRT2, FS_IN_TexCoord).rgb);
}

// ------------------------------------------------------------------

vec4 unpack_albedo()
{
    return vec4(texture(s_GBufferRT1, FS_IN_TexCoord).xyz, 1.0);
}

// ------------------------------------------------------------------

float unpack_metalness()
{
    return texture(s_GBufferRT4, FS_IN_TexCoord).x;
}

// ------------------------------------------------------------------

float unpack_roughness()
{
    return texture(s_GBufferRT4, FS_IN_TexCoord).y;
}

// ------------------------------------------------------------------

float unpack_depth()
{
	return texture(s_Depth, FS_IN_TexCoord).x;
}

// ------------------------------------------------------------------

float unpack_ssao()
{
	return texture(s_SSAO, FS_IN_TexCoord).x;
}

// ------------------------------------------------------------------

void fill_fragment_properties(inout FragmentProperties f)
{
	f.FragDepth = unpack_depth();
	f.Position = world_position_from_depth(FS_IN_TexCoord, f.FragDepth);
	f.TexCoords = FS_IN_TexCoord;
}

// ------------------------------------------------------------------

void fragment_func(inout MaterialProperties m)
{
	m.albedo = unpack_albedo();
	m.normal = unpack_normal();
	m.metallic = unpack_metalness();
	m.roughness = unpack_roughness();
}

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
	FragmentProperties f;

	fill_fragment_properties(f);

	MaterialProperties m;

	// Set material properties
	fragment_func(m);

	PBRProperties pbr;

	// Set PBR properties
	pbr.N = m.normal;
	pbr.V = normalize(view_pos.xyz - f.Position); // FragPos -> ViewPos vector
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

	// Add all light contributions
	Lo += pbr_light_contribution(m, f, pbr);

	vec3 irradiance = texture(s_IrradianceMap, pbr.N).rgb;
	vec3 diffuse = irradiance * m.albedo.xyz;

	// Sample prefilter map and BRDF LUT
	// vec3 prefilteredColor = textureLod(s_PrefilteredMap, pbr.R, m.roughness * kMaxLOD).rgb;
	// vec2 brdf = texture(s_BRDF, vec2(max(pbr.NdotV, 0.0), m.roughness)).rg;
	// vec3 specular = prefilteredColor * (pbr.F * brdf.x + brdf.y);

	// vec3 ambient = (pbr.kD * diffuse + specular) * kAmbient;
	float ambient = unpack_ssao();
	vec3 color = Lo + (m.albedo.xyz * ambient * 0.3);// + ambient;

    FS_OUT_Color = color;
}

// ------------------------------------------------------------------