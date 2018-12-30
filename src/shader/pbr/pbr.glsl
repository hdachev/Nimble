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
	vec3 L = normalize(-directionalLight.direction.xyz); // FragPos -> LightPos vector
	vec3 H = normalize(pbr.V + L);
	float HdotV = clamp(dot(H, pbr.V), 0.0, 1.0);
	float NdotH = max(dot(pbr.N, H), 0.0);
	float NdotL = max(dot(pbr.N, pbr.L), 0.0);

	// Shadows ------------------------------------------------------------------
	float frag_depth = f.FragDepth;

#ifdef DIRECTIONAL_LIGHT_SHADOWS
	float shadow = directional_light_shadows(frag_depth, f.Position, f.Normal, L);
	vec3 shadow_debug = debug_color(frag_depth);
#else
	float shadow = 1.0;
#endif

	// Radiance -----------------------------------------------------------------
	vec3 Li = directionalLight.color.xyz * directionalLight.color.w;
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

	return shadow * (kD * m.albedo.xyz / kPI + specular) * Li * NdotL;
}

// ------------------------------------------------------------------

vec3 pbr_point_lights(in MaterialProperties m, in FragmentProperties f,  in PBRProperties pbr)
{
	vec3 Lo = vec3(0.0);
	vec3 shadow_debug = vec3(0.0);

	for (int i = 0; i < pointLightCount; i++)
	{
		vec3 L = normalize(pointLights[i].position.xyz - f.Position); // FragPos -> LightPos vector
		vec3 H = normalize(pbr.V + L);
		float HdotV = clamp(dot(H, pbr.V), 0.0, 1.0);
		float NdotH = max(dot(pbr.N, H), 0.0);
		float NdotL = max(dot(pbr.N, L), 0.0);

		// Shadows ------------------------------------------------------------------
		float frag_depth = f.FragDepth;

#ifdef POINT_LIGHT_SHADOWS
		float shadow = point_light_shadows();
		shadow_debug = debug_color(frag_depth);
#else
		float shadow = 1.0;
#endif

		// Radiance -----------------------------------------------------------------
		float distance = length(pointLights[i].position.xyz - f.Position);
		float attenuation = 1.0 / (distance * distance);
		vec3 Li = pointLights[i].color.xyz * attenuation;
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
		Lo += shadow * (pbr.kD * m.albedo.xyz / kPI + specular) * Li * NdotL;
		// --------------------------------------------------------------------------
	}

	return Lo;
}

// ------------------------------------------------------------------

vec3 pbr_spot_lights(in MaterialProperties m, in FragmentProperties f,  in PBRProperties pbr)
{
	vec3 Lo = vec3(0.0);

	return Lo;
}

// ------------------------------------------------------------------