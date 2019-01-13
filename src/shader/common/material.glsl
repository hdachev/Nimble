// ------------------------------------------------------------------
// MATERIAL ---------------------------------------------------------
// ------------------------------------------------------------------

#define MIN_ROUGHNESS 0.04

float convert_metallic(vec3 diffuse, vec3 specular, float maxSpecular) 
{
	float perceivedDiffuse = sqrt(0.299 * diffuse.r * diffuse.r + 0.587 * diffuse.g * diffuse.g + 0.114 * diffuse.b * diffuse.b);
	float perceivedSpecular = sqrt(0.299 * specular.r * specular.r + 0.587 * specular.g * specular.g + 0.114 * specular.b * specular.b);

	if (perceivedSpecular < MIN_ROUGHNESS)
		return 0.0;

	float a = MIN_ROUGHNESS;
	float b = perceivedDiffuse * (1.0 - maxSpecular) / (1.0 - MIN_ROUGHNESS) + perceivedSpecular - 2.0 * MIN_ROUGHNESS;
	float c = MIN_ROUGHNESS - perceivedSpecular;
	float D = max(b * b - 4.0 * a * c, 0.0);
	return clamp((-b + sqrt(D)) / (2.0 * a), 0.0, 1.0);
}

// ------------------------------------------------------------------

vec4 get_albedo(vec2 tex_coord)
{
#ifdef TEXTURE_ALBEDO
	return texture(s_Albedo, tex_coord);
#else
#ifdef UNIFORM_ALBEDO
	return u_Albedo;
#else
	return vec4(1.0);
#endif
#endif
}

// ------------------------------------------------------------------

vec3 get_normal(in FragmentProperties f)
{
#ifdef TEXTURE_NORMAL
	return get_normal_from_map(f.Tangent, f.Bitangent, f.Normal, f.TexCoords, s_Normal);
#else
	return f.Normal;
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
#else 
#ifdef UNIFORM_METAL_SPEC
	return u_MetalRough.y;
#else
	return 0.0;
#endif
#endif
}

// ------------------------------------------------------------------

float get_metallic(vec2 tex_coord)
{
#ifdef TEXTURE_METAL_SPEC
	return texture(s_MetalSpec, tex_coord).x;
#else
#ifdef UNIFORM_METAL_SPEC
	return u_MetalRough.x;
#else
	return 0.0;
#endif
#endif
}

// ------------------------------------------------------------------