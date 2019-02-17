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
	return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(1.0 - HdotV, 5.0);
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

vec3 pbr_directional_lights(in MaterialProperties m, in FragmentProperties f,  in PBRProperties pbr)
{
	vec3 Lo = vec3(0.0);

#ifdef CSM_DEBUG
	vec3 shadow_debug = vec3(0.0);
#endif

#ifdef DIRECTIONAL_LIGHTS
	int shadow_casting_light_idx = 0;

	for (int i = 0; i < directional_light_count; i++)
	{
		vec3 L = normalize(-directional_light_direction[i].xyz); // FragPos -> LightPos vector
		vec3 H = normalize(pbr.V + L);
		float HdotV = clamp(dot(H, pbr.V), 0.0, 1.0);
		float NdotH = max(dot(pbr.N, H), 0.0);
		float NdotL = max(dot(pbr.N, L), 0.0);

		// Shadows ------------------------------------------------------------------
		float frag_depth = f.FragDepth;
		float visibility = 1.0;

	#ifdef DIRECTIONAL_LIGHT_SHADOW_MAPPING
		if (directional_light_casts_shadow[i] == 1)
		{
			visibility = directional_light_shadows(f, shadow_casting_light_idx, i);
			shadow_casting_light_idx++;
		}
	#ifdef CSM_DEBUG
		shadow_debug += debug_color(frag_depth);
	#endif
	#endif

		// Radiance -----------------------------------------------------------------
		vec3 Li = directional_light_color_intensity[i].xyz * directional_light_color_intensity[i].w;
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
	}

#endif

#ifdef CSM_DEBUG
	Lo *= shadow_debug;
#endif 

	return Lo;
}

// ------------------------------------------------------------------

vec3 pbr_point_lights(in MaterialProperties m, in FragmentProperties f,  in PBRProperties pbr)
{
	vec3 Lo = vec3(0.0);

#ifdef POINT_LIGHTS
	int shadow_casting_light_idx = 0;

	for (int i = 0; i < point_light_count; i++)
	{
		vec3 L = normalize(point_light_position_range[i].xyz - f.Position); // FragPos -> LightPos vector
		vec3 H = normalize(pbr.V + L);
		float HdotV = clamp(dot(H, pbr.V), 0.0, 1.0);
		float NdotH = max(dot(pbr.N, H), 0.0);
		float NdotL = max(dot(pbr.N, L), 0.0);

		// Shadows ------------------------------------------------------------------
		float frag_depth = f.FragDepth;
		float visibility = 1.0;

#ifdef POINT_LIGHT_SHADOW_MAPPING
		if (point_light_casts_shadow[i] == 1)
		{
			visibility = point_light_shadows(f, shadow_casting_light_idx, i);	
			shadow_casting_light_idx++;
		}
#endif

		// Radiance -----------------------------------------------------------------
		float distance = length(point_light_position_range[i].xyz - f.Position);
		float attenuation = smoothstep(point_light_position_range[i].w, 0, distance);
		vec3 Li = point_light_color_intensity[i].xyz * point_light_color_intensity[i].w * attenuation;
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
	}

#endif

	return Lo;
}


// ------------------------------------------------------------------

vec3 pbr_spot_lights(in MaterialProperties m, in FragmentProperties f,  in PBRProperties pbr)
{
	vec3 Lo = vec3(0.0);

#ifdef SPOT_LIGHTS
	int shadow_casting_light_idx = 0;

	for (int i = 0; i < spot_light_count; i++)
	{
		vec3 L = normalize(spot_light_position[i].xyz - f.Position); // FragPos -> LightPos vector
		vec3 H = normalize(pbr.V + L);
		float HdotV = clamp(dot(H, pbr.V), 0.0, 1.0);
		float NdotH = max(dot(pbr.N, H), 0.0);
		float NdotL = max(dot(pbr.N, L), 0.0);

		// Shadows ------------------------------------------------------------------
		float frag_depth = f.FragDepth;
		float visibility = 1.0;

#ifdef SPOT_LIGHT_SHADOW_MAPPING
		if (spot_light_casts_shadow[i] == 1)
		{
			visibility = spot_light_shadows(f.Position, shadow_casting_light_idx, i);	
			shadow_casting_light_idx++;
		}
#endif

		// Radiance -----------------------------------------------------------------
		float theta = dot(L, normalize(-spot_light_direction_range[i].xyz));
		float distance = length(spot_light_position[i].xyz - f.Position);
		float inner_cut_off = spot_light_cutoff_inner_outer[i].x;
		float outer_cut_off = spot_light_cutoff_inner_outer[i].y;
		float epsilon = inner_cut_off - outer_cut_off;
		float attenuation = smoothstep(spot_light_direction_range[i].w, 0, distance) * clamp((theta - outer_cut_off) / epsilon, 0.0, 1.0) * visibility;
		vec3 Li = spot_light_color_intensity[i].xyz * spot_light_color_intensity[i].w * attenuation;
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
	}

#endif

	return Lo;
}

// ------------------------------------------------------------------ 

vec3 pbr_light_contribution(in MaterialProperties m, in FragmentProperties f,  in PBRProperties pbr)
{
	vec3 Lo = vec3(0.0);

	Lo += pbr_directional_lights(m, f, pbr);
	Lo += pbr_point_lights(m, f, pbr);
	Lo += pbr_spot_lights(m, f, pbr);

	return Lo;
}

// ------------------------------------------------------------------