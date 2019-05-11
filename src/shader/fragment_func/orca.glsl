void fragment_func(inout MaterialProperties m, inout FragmentProperties f)
{
#ifdef MATERIAL_GLSL
	m.albedo = get_albedo(f.TexCoords);
	m.normal = get_normal(f);
#ifdef TEXTURE_METAL_SPEC
	m.metallic = texture(s_MetalSpec, f.TexCoords).b;
	m.roughness = texture(s_MetalSpec, f.TexCoords).g;
#endif
#endif
}