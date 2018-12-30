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

#ifdef NORMAL_TEXTURE
	in vec3 PS_IN_Tangent;
	in vec3 PS_IN_Bitangent;
#endif

#ifdef DISPLACEMENT_TYPE_PARALLAX_OCCLUSION
	in vec3 PS_IN_TangentViewPos;
	in vec3 PS_IN_TangentFragPos;
#endif

#include <../csm/csm.glsl>
#include <../common/helper.glsl>
#include <../common/material.glsl>
#include <../pbr/pbr.glsl>

// ------------------------------------------------------------------
// OUTPUT VARIABLES  ------------------------------------------------
// ------------------------------------------------------------------

layout (location = 0) out vec3 PS_OUT_Color;

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
}

// ------------------------------------------------------------------

#ifndef FRAGMENT_SHADER_FUNC
#define FRAGMENT_SHADER_FUNC

void fragment_func(inout MaterialProperties m, inout FragmentProperties f, vec2 tex_coord)
{
	m.albedo = get_albedo(tex_coord);
	m.normal = get_normal(tex_coord, f);
	m.metallic = get_metallic(tex_coord);
	m.roughness = get_roughness(tex_coord);
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

	// Calculate texture coordinates
	vec2 tex_coord = get_texcoords(PS_IN_TexCoord);

	// Set material properties
	fragment_func(m, f, tex_coord);
}

// ------------------------------------------------------------------