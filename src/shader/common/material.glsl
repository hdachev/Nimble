// ------------------------------------------------------------------
// MATERIAL ---------------------------------------------------------
// ------------------------------------------------------------------

float convert_metallic(vec3 diffuse, vec3 specular, float maxSpecular) 
{
	float perceivedDiffuse = sqrt(0.299 * diffuse.r * diffuse.r + 0.587 * diffuse.g * diffuse.g + 0.114 * diffuse.b * diffuse.b);
	float perceivedSpecular = sqrt(0.299 * specular.r * specular.r + 0.587 * specular.g * specular.g + 0.114 * specular.b * specular.b);

	if (perceivedSpecular < c_MinRoughness)
		return 0.0;

	float a = c_MinRoughness;
	float b = perceivedDiffuse * (1.0 - maxSpecular) / (1.0 - c_MinRoughness) + perceivedSpecular - 2.0 * c_MinRoughness;
	float c = c_MinRoughness - perceivedSpecular;
	float D = max(b * b - 4.0 * a * c, 0.0);
	return clamp((-b + sqrt(D)) / (2.0 * a), 0.0, 1.0);
}

// ------------------------------------------------------------------

vec4 get_albedo(vec2 tex_coord)
{
#ifdef TEXTURE_ALBEDO
	return texture(s_Albedo, tex_coord);
#elif UNIFORM_ALBEDO
	return u_Albedo;
#endif
}

// ------------------------------------------------------------------

vec3 get_normal(vec2 tex_coord, in FragmentProperties f)
{
#ifdef NORMAL_TEXTURE
	return get_normal_from_map(f.Tangent, f.Bitangent, f.Normal, tex_coord, s_Normal);
#else
	return v.Normal;
#endif
}

// ------------------------------------------------------------------

float get_roughness(vec2 tex_coord)
{
#ifdef TEXTURE_ROUGH_SMOOTH
	#ifdef SPECULAR_WORKFLOW
		vec3 specular = texture(s_RoughSmooth, tex_coord).xyz;
		float maxSpecular = max(max(specular.r, specular.g), specular.b);
		return convert_metallic(get_albedo(tex_coord).xyz, specular, maxSpecular);
	#else
		return texture(s_RoughSmooth, tex_coord).x;
	#endif
#elif UNIFORM_METAL_SPEC
	return s_MetalSpec.y;
#endif
}

// ------------------------------------------------------------------

float get_metallic(vec2 tex_coord)
{
#ifdef TEXTURE_METAL_SPEC
	return texture(s_MetalSpec, tex_coord).x;
#elif UNIFORM_METAL_SPEC
	return s_MetalSpec.x;
#endif
}

// ------------------------------------------------------------------