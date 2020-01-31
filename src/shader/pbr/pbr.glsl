// ------------------------------------------------------------------
// CONSTANTS  -------------------------------------------------------
// ------------------------------------------------------------------

const float kPI 	   = 3.14159265359;
const float kMaxLOD    = 6.0;
const float kAmbient   = 1.0;

// ------------------------------------------------------------------
// PBR FUNCTIONS ----------------------------------------------------
// ------------------------------------------------------------------

vec3 fresnel_schlick_roughness(float HdotV, vec3 F0, float roughness)
{
	return clamp(F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(1.0 - HdotV, 5.0), 0.0, 1.0);
}

// ------------------------------------------------------------------

float distribution_trowbridge_reitz_ggx(float NdotH, float roughness)
{
	// a = Roughness
	float a = roughness * roughness;
	float a2 = a * a;

	float numerator = a2;
	float denominator = ((NdotH * NdotH) * (a2 - 1.0) + 1.0);
	denominator = kPI * denominator * denominator;

	return numerator / denominator;
}

// ------------------------------------------------------------------

float geometry_schlick_ggx(float costTheta, float roughness)
{
	float r = (roughness + 1.0);
	float k = (r*r) / 8.0;

	float numerator = costTheta;
	float denominator = costTheta * (1.0 - k) + k;
	return numerator / denominator;
}

// ------------------------------------------------------------------

float geometry_smith(float NdotV, float NdotL, float roughness)
{
	float G1 = geometry_schlick_ggx(NdotV, roughness);
	float G2 = geometry_schlick_ggx(NdotL, roughness);
	return G1 * G2;
}

// ------------------------------------------------------------------

vec3 pbr_directional_light_contribution(in MaterialProperties m, 
								  		in FragmentProperties f,  
								  		in PBRProperties pbr,
								  		int i)
{
	vec3 Lo = vec3(0.0);

	vec3 L = normalize(-directional_light_direction(i)); // FragPos -> LightPos vector
	vec3 H = normalize(pbr.V + L);
	float HdotV = clamp(dot(H, pbr.V), 0.0, 1.0);
	float NdotH = max(dot(pbr.N, H), 0.0);
	float NdotL = max(dot(pbr.N, L), 0.0);

	// Shadows ------------------------------------------------------------------
	float frag_depth = f.FragDepth;
	float visibility = 1.0;

#ifdef DIRECTIONAL_LIGHT_SHADOW_MAPPING
	if (directional_light_first_shadow_map_index(i) >= 0)
		visibility = directional_light_shadows(f, i);

	#ifdef CSM_DEBUG
		shadow_debug += csm_debug_color(frag_depth, shadow_map_idx);
	#endif
#endif

	// Radiance -----------------------------------------------------------------
	vec3 Li = directional_light_color(i) * directional_light_intensity(i);
	// --------------------------------------------------------------------------

	// Specular Term ------------------------------------------------------------
	float D = distribution_trowbridge_reitz_ggx(NdotH, m.roughness);
	float G = geometry_smith(pbr.NdotV, NdotL, m.roughness);

	vec3 numerator = D * G * pbr.F;
	float denominator = 4.0 * pbr.NdotV * NdotL; 

	vec3 specular = numerator / max(denominator, 0.001);
	// --------------------------------------------------------------------------

	// Diffuse Term -------------------------------------------------------------
	vec3 diffuse = m.albedo.xyz / kPI;
	// --------------------------------------------------------------------------

	Lo += visibility * (pbr.kD * m.albedo.xyz / kPI + specular) * Li * NdotL;

	return Lo;
}

// ------------------------------------------------------------------

vec3 pbr_point_light_contribution(in MaterialProperties m, 
								  in FragmentProperties f,  
								  in PBRProperties pbr,
								  int i)
{
	vec3 Lo = vec3(0.0);

	vec3 L = normalize(point_light_position(i) - f.Position); // FragPos -> LightPos vector
	vec3 H = normalize(pbr.V + L);
	float HdotV = clamp(dot(H, pbr.V), 0.0, 1.0);
	float NdotH = max(dot(pbr.N, H), 0.0);
	float NdotL = max(dot(pbr.N, L), 0.0);

	// Shadows ------------------------------------------------------------------
	float frag_depth = f.FragDepth;
	float visibility = 1.0;

#ifdef POINT_LIGHT_SHADOW_MAPPING
	if (point_light_shadow_map_index(i) >= 0)
		visibility = point_light_shadows(f, i);	
#endif

	// Radiance -----------------------------------------------------------------
	float distance = length(point_light_position(i) - f.Position);
	float attenuation = smoothstep(point_light_far_field(i), 0, distance);
	vec3 Li = point_light_color(i) * point_light_intensity(i) * attenuation;
	// --------------------------------------------------------------------------

	// Specular Term ------------------------------------------------------------
	float D = distribution_trowbridge_reitz_ggx(NdotH, m.roughness);
	float G = geometry_smith(pbr.NdotV, NdotL, m.roughness);

	vec3 numerator = D * G * pbr.F;
	float denominator = 4.0 * pbr.NdotV * NdotL; 

	vec3 specular = numerator / max(denominator, 0.001);
	// --------------------------------------------------------------------------

	// Diffuse Term -------------------------------------------------------------
	vec3 diffuse = m.albedo.xyz / kPI;
	// --------------------------------------------------------------------------

	// Combination --------------------------------------------------------------
	Lo += visibility * (pbr.kD * m.albedo.xyz / kPI + specular) * Li * NdotL;
	// --------------------------------------------------------------------------

	return Lo;
}

// ------------------------------------------------------------------

vec3 pbr_spot_light_contribution(in MaterialProperties m, 
								 in FragmentProperties f,  
								 in PBRProperties pbr,
								 int i)
{
	vec3 Lo = vec3(0.0);

	vec3 L = normalize(spot_light_position(i) - f.Position); // FragPos -> LightPos vector
	vec3 H = normalize(pbr.V + L);
	float HdotV = clamp(dot(H, pbr.V), 0.0, 1.0);
	float NdotH = max(dot(pbr.N, H), 0.0);
	float NdotL = max(dot(pbr.N, L), 0.0);

	// Shadows ------------------------------------------------------------------
	float frag_depth = f.FragDepth;
	float visibility = 1.0;

#ifdef SPOT_LIGHT_SHADOW_MAPPING
	if (spot_light_shadow_map_index(i) >= 0)
		visibility = spot_light_shadows(f, i);	
#endif

	// Radiance -----------------------------------------------------------------
	float theta = dot(L, normalize(-spot_light_direction(i)));
	float distance = length(spot_light_position(i) - f.Position);
	float inner_cut_off = spot_light_inner_cutoff(i);
	float outer_cut_off = spot_light_outer_cutoff(i);
	float epsilon = inner_cut_off - outer_cut_off;
	float attenuation = smoothstep(spot_light_range(i), 0, distance) * clamp((theta - outer_cut_off) / epsilon, 0.0, 1.0) * visibility;
	vec3 Li = spot_light_color(i) * spot_light_intensity(i) * attenuation;
	// --------------------------------------------------------------------------

	// Specular Term ------------------------------------------------------------
	float D = distribution_trowbridge_reitz_ggx(NdotH, m.roughness);
	float G = geometry_smith(pbr.NdotV, NdotL, m.roughness);

	vec3 numerator = D * G * pbr.F;
	float denominator = 4.0 * pbr.NdotV * NdotL; 

	vec3 specular = numerator / max(denominator, 0.001);
	// --------------------------------------------------------------------------

	// Diffuse Term -------------------------------------------------------------
	vec3 diffuse = m.albedo.xyz / kPI;
	// --------------------------------------------------------------------------

	// Combination --------------------------------------------------------------
	Lo += (pbr.kD * m.albedo.xyz / kPI + specular) * Li * NdotL;
	// --------------------------------------------------------------------------

	return Lo;
}

// ------------------------------------------------------------------ 

vec3 pbr_light_contribution(in MaterialProperties m, in FragmentProperties f,  in PBRProperties pbr)
{
	vec3 Lo = vec3(0.0);

	for (int i = 0; i < light_count.x; i++)
	{
		int type = light_type(i);

		if (type == LIGHT_TYPE_DIRECTIONAL)
		{
		#ifdef DIRECTIONAL_LIGHTS
			Lo += pbr_directional_light_contribution(m, f, pbr, i);
		#endif
		}
		else if (type == LIGHT_TYPE_SPOT)
		{
		#ifdef SPOT_LIGHTS
			Lo += pbr_spot_light_contribution(m, f, pbr, i);
		#endif
		}
		else if (type == LIGHT_TYPE_POINT)
		{
		#ifdef POINT_LIGHTS
			Lo += pbr_point_light_contribution(m, f, pbr, i);
		#endif
		}
	}

	return Lo;
}

// ------------------------------------------------------------------