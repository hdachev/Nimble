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