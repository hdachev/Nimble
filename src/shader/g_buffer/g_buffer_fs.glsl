#include <../common/uniforms.glsl>

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

// ------------------------------------------------------------------
// OUTPUT VARIABLES  ------------------------------------------------
// ------------------------------------------------------------------

layout (location = 0) out vec4 FS_OUT_Albedo;
layout (location = 1) out vec4 FS_OUT_Normal;
layout (location = 2) out vec4 FS_OUT_Velocity;
layout (location = 3) out vec4 FS_OUT_MetalRough;

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

	FS_OUT_Albedo = m.albedo;
	FS_OUT_Normal = vec4(m.normal, 0.0);
	FS_OUT_MetalRough = vec4(m.metallic, m.roughness, 0.0, 0.0);
	FS_OUT_Velocity = vec4(motion_vector(PS_IN_LastScreenPosition, PS_IN_ScreenPosition), 0.0, 0.0);
}

// ------------------------------------------------------------------