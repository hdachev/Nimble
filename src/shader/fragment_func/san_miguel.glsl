void fragment_func(inout MaterialProperties m, inout FragmentProperties f)
{
#ifdef MATERIAL_GLSL
	m.albedo = get_albedo(f.TexCoords);

#ifdef TEXTURE_NORMAL
	vec2 normal_map_rg = texture(s_Normal, f.TexCoords).ga;

	vec3 reconstructed_normal;
	reconstructed_normal.xy = normal_map_rg * 2.0 - 1.0;
	reconstructed_normal.z = sqrt(1.0 - dot(reconstructed_normal.xy, reconstructed_normal.xy));

	m.normal = get_normal_ex(f, reconstructed_normal);
#else
	m.normal = f.Normal;
#endif

#ifdef TEXTURE_METAL_SPEC
	m.metallic = texture(s_MetalSpec, f.TexCoords).b;
	m.roughness = texture(s_MetalSpec, f.TexCoords).a;
#endif
#endif
}