#include <../common/material.glsl>

#ifndef FRAGMENT_SHADER_FUNC
#define FRAGMENT_SHADER_FUNC

void fragment_func(inout MaterialProperties m, inout FragmentProperties f)
{
	m.albedo = get_albedo(f.TexCoords);
	m.normal = get_normal(f);
	m.metallic = texture(s_MetalSpec, f.TexCoords).b;
	m.roughness = texture(s_MetalSpec, f.TexCoords).g;
}

#endif