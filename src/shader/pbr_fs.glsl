// ------------------------------------------------------------------
// STRUCTURES -------------------------------------------------------
// ------------------------------------------------------------------

#define MAX_POINT_LIGHTS 32

struct PointLight
{
	vec4 position;
	vec4 color;
};

struct DirectionalLight
{
	vec4 direction;
	vec4 color;
};

// ------------------------------------------------------------------
// UNIFORM BUFFERS --------------------------------------------------
// ------------------------------------------------------------------

#define MAX_SHADOW_FRUSTUM 8

struct ShadowFrustum
{
	mat4  shadowMatrix;
	float farPlane;
};

layout (std140) uniform u_PerFrame //#binding 0
{ 
	mat4 		  lastViewProj;
	mat4 		  viewProj;
	mat4 		  invViewProj;
	mat4 		  projMat;
	mat4 		  viewMat;
	vec4 		  viewPos;
	vec4 		  viewDir;
	int			  numCascades;
	ShadowFrustum shadowFrustums[MAX_SHADOW_FRUSTUM];
};

layout (std140) uniform u_PerEntity //#binding 1
{
	mat4 mvpMat;
	mat4 lastMvpMat;
	mat4 modelMat;	
	vec4 worldPos;
};

layout (std140) uniform u_PerScene //#binding 2
{
	PointLight 		 pointLights[MAX_POINT_LIGHTS];
	DirectionalLight directionalLight;
	int				 pointLightCount;
};

// ------------------------------------------------------------------
// CONSTANTS  -------------------------------------------------------
// ------------------------------------------------------------------

const float kPI 	   = 3.14159265359;
const float kMaxLOD    = 6.0;

// ------------------------------------------------------------------
// SAMPLERS  --------------------------------------------------------
// ------------------------------------------------------------------

uniform sampler2D s_Albedo; //#slot 0
uniform sampler2D s_Normal; //#slot 1
uniform sampler2D s_Metalness; //#slot 2
uniform sampler2D s_Roughness; //#slot 3
uniform samplerCube s_IrradianceMap; //#slot 4
uniform samplerCube s_PrefilteredMap; //#slot 5
uniform sampler2D s_BRDF; //#slot 6
uniform sampler2DArray s_ShadowMap; //#slot 7

// ------------------------------------------------------------------
// INPUT VARIABLES  -------------------------------------------------
// ------------------------------------------------------------------

in vec3 PS_IN_Position;
in vec4 PS_IN_NDCFragPos;
in vec3 PS_IN_CamPos;
in vec3 PS_IN_Normal;
in vec2 PS_IN_TexCoord;
in vec3 PS_IN_Tangent;
in vec3 PS_IN_Bitangent;

// ------------------------------------------------------------------
// OUTPUT VARIABLES  ------------------------------------------------
// ------------------------------------------------------------------

layout (location = 0) out vec4 PS_OUT_Color;

// ------------------------------------------------------------------
// FUNCTIONS --------------------------------------------------------
// ------------------------------------------------------------------

float depth_compare(float a, float b, float bias)
{
    return a - bias > b ? 1.0 : 0.0;
}

float shadow_occlussion(float frag_depth, vec3 n, vec3 l)
{
	int index = 0;
    float blend = 0.0;
    
	// Find shadow cascade.
	for (int i = 0; i < numCascades - 1; i++)
	{
		if (frag_depth > shadowFrustums[i].farPlane)
			index = i + 1;
	}

	blend = clamp( (frag_depth - shadowFrustums[index].farPlane * 0.995) * 200.0, 0.0, 1.0);
    
    // Apply blend options.
    //blend *= options.z;

	// Transform frag position into Light-space.
	vec4 light_space_pos = shadowFrustums[index].shadowMatrix * vec4(PS_IN_Position, 1.0f);

	float current_depth = light_space_pos.z;
    
	float bias = max(0.0005 * (1.0 - dot(n, l)), 0.0005);  

	float shadow = 0.0;
	vec2 texelSize = 1.0 / textureSize(s_ShadowMap, 0).xy;
	for(int x = -1; x <= 1; ++x)
	{
	    for(int y = -1; y <= 1; ++y)
	    {
	        float pcfDepth = texture(s_ShadowMap, vec3(light_space_pos.xy + vec2(x, y) * texelSize, float(index))).r; 
	        shadow += current_depth - bias > pcfDepth ? 1.0 : 0.0;        
	    }    
	}
	shadow /= 9.0;
	
	return (1.0 - shadow);
    // if (options.x == 1.0)
    // {
    //     //if (blend > 0.0 && index != num_cascades - 1)
    //     //{
    //     //    light_space_pos = texture_matrices[index + 1] * vec4(PS_IN_WorldFragPos, 1.0f);
    //     //    shadow_map_depth = texture(s_ShadowMap, vec3(light_space_pos.xy, float(index + 1))).r;
    //     //    current_depth = light_space_pos.z;
    //     //    float next_shadow = depth_compare(current_depth, shadow_map_depth, bias);
    //     //    
    //     //    return (1.0 - blend) * shadow + blend * next_shadow;
    //     //}
    //     //else
	// 		return (1.0 - shadow);
    // }
    // else
    //     return 0.0;
}

vec3 debug_color(float frag_depth)
{
	int index = 0;

	// Find shadow cascade.
	for (int i = 0; i < numCascades - 1; i++)
	{
		if (frag_depth > shadowFrustums[index].farPlane)
			index = i + 1;
	}

	if (index == 0)
		return vec3(1.0, 0.0, 0.0);
	else if (index == 1)
		return vec3(0.0, 1.0, 0.0);
	else if (index == 2)
		return vec3(0.0, 0.0, 1.0);
	else
		return vec3(1.0, 1.0, 0.0);
}

vec3 getNormalFromMap()
{
	// Create TBN matrix.
    mat3 TBN = mat3(normalize(PS_IN_Tangent), normalize(PS_IN_Bitangent), normalize(PS_IN_Normal));

    // Sample tangent space normal vector from normal map and remap it from [0, 1] to [-1, 1] range.
    vec3 n = normalize(texture(s_Normal, PS_IN_TexCoord).xyz * 2.0 - 1.0);

    // Multiple vector by the TBN matrix to transform the normal from tangent space to world space.
    n = normalize(TBN * n);

    return n;
}

vec3 FresnelSchlickRoughness(float HdotV, vec3 F0, float roughness)
{
	return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(1.0 - HdotV, 5.0);
}

float DistributionTrowbridgeReitzGGX(float NdotH, float roughness)
{
	// a = Roughness
	float a = roughness * roughness;
	float a2 = a * a;

	float numerator = a2;
	float denominator = ((NdotH * NdotH) * (a2 - 1.0) + 1.0);
	denominator = kPI * denominator * denominator;

	return numerator / denominator;
}

float GeometrySchlickGGX(float costTheta, float roughness)
{
	float r = (roughness + 1.0);
	float k = (r*r) / 8.0;

	float numerator = costTheta;
	float denominator = costTheta * (1.0 - k) + k;
	return numerator / denominator;
}

float GeometrySmith(float NdotV, float NdotL, float roughness)
{
	float G1 = GeometrySchlickGGX(NdotV, roughness);
	float G2 = GeometrySchlickGGX(NdotL, roughness);
	return G1 * G2;
}

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
	const float kAmbient   = 1.0;

	vec4 kAlbedoAlpha = texture(s_Albedo, PS_IN_TexCoord);
	float kMetalness = texture(s_Metalness, PS_IN_TexCoord).x; 
	float kRoughness = texture(s_Roughness, PS_IN_TexCoord).x; 

	if (kAlbedoAlpha.w < 0.5)
		discard;

	vec3 kAlbedo = kAlbedoAlpha.xyz;

	vec3 N = getNormalFromMap();
	vec3 V = normalize(PS_IN_CamPos - PS_IN_Position); // FragPos -> ViewPos vector
	vec3 R = reflect(-V, N); 
	vec3 Lo = vec3(0.0);

	vec3 F0 = vec3(0.04);
	F0 = mix(F0, kAlbedo, kMetalness);

	float NdotV = max(dot(N, V), 0.0);
	vec3  F = FresnelSchlickRoughness(NdotV, F0, kRoughness);

	vec3 shadow_debug = vec3(0.0);

	// For each directional light...
	{
		vec3 L = normalize(-directionalLight.direction.xyz); // FragPos -> LightPos vector
		vec3 H = normalize(V + L);
		float HdotV = clamp(dot(H, V), 0.0, 1.0);
		float NdotH = max(dot(N, H), 0.0);
		float NdotL = max(dot(N, L), 0.0);

		// Shadows ------------------------------------------------------------------
		float frag_depth = (PS_IN_NDCFragPos.z / PS_IN_NDCFragPos.w) * 0.5 + 0.5;
		float shadow = shadow_occlussion(frag_depth, PS_IN_Normal, L);

		shadow_debug = debug_color(frag_depth);

		// Radiance -----------------------------------------------------------------

		vec3 Li = directionalLight.color.xyz * directionalLight.color.w;

		// --------------------------------------------------------------------------

		// Specular Term ------------------------------------------------------------
		float D = DistributionTrowbridgeReitzGGX(NdotH, kRoughness);
		float G = GeometrySmith(NdotV, NdotL, kRoughness);

		vec3 numerator = D * G * F;
		float denominator = 4.0 * NdotV * NdotL; 

		vec3 specular = numerator / max(denominator, 0.001);
		// --------------------------------------------------------------------------

		// Diffuse Term -------------------------------------------------------------
		vec3 diffuse = kAlbedo / kPI;
		// --------------------------------------------------------------------------

		// Combination --------------------------------------------------------------
		vec3 kS = F;
		vec3 kD = vec3(1.0) - kS;
		kD *= 1.0 - kMetalness;

		Lo += shadow * (kD * kAlbedo / kPI + specular) * Li * NdotL;
		// --------------------------------------------------------------------------
	}

	// For each point light...
	for (int i = 0; i < pointLightCount; i++)
	{
		vec3 L = normalize(pointLights[i].position.xyz - PS_IN_Position); // FragPos -> LightPos vector
		vec3 H = normalize(V + L);
		float HdotV = clamp(dot(H, V), 0.0, 1.0);
		float NdotH = max(dot(N, H), 0.0);
		float NdotL = max(dot(N, L), 0.0);

		// Radiance -----------------------------------------------------------------

		float distance = length(pointLights[i].position.xyz - PS_IN_Position);
		float attenuation = 1.0 / (distance * distance);
		vec3 Li = pointLights[i].color.xyz * attenuation;

		// --------------------------------------------------------------------------

		// Specular Term ------------------------------------------------------------
		float D = DistributionTrowbridgeReitzGGX(NdotH, kRoughness);
		float G = GeometrySmith(NdotV, NdotL, kRoughness);

		vec3 numerator = D * G * F;
		float denominator = 4.0 * NdotV * NdotL; 

		vec3 specular = numerator / max(denominator, 0.001);
		// --------------------------------------------------------------------------

		// Diffuse Term -------------------------------------------------------------
		vec3 diffuse = kAlbedo / kPI;
		// --------------------------------------------------------------------------

		// Combination --------------------------------------------------------------
		vec3 kS = F;
		vec3 kD = vec3(1.0) - kS;
		kD *= 1.0 - kMetalness;

		Lo += (kD * kAlbedo / kPI + specular) * Li * NdotL;
		// --------------------------------------------------------------------------
	}

	vec3 kS = F;
	vec3 kD = 1.0 - kS;
	kD *= 1.0 - kMetalness;

	vec3 irradiance = texture(s_IrradianceMap, N).rgb;
	vec3 diffuse = irradiance * kAlbedo;

	// Sample prefilter map and BRDF LUT
	vec3 prefilteredColor = textureLod(s_PrefilteredMap, R, kRoughness * kMaxLOD).rgb;
	vec2 brdf = texture(s_BRDF, vec2(max(NdotV, 0.0), kRoughness)).rg;
	vec3 specular = prefilteredColor * (F * brdf.x + brdf.y);

	vec3 ambient = (kD * diffuse + specular) * kAmbient;
	// vec3 ambient = vec3(0.03) * diffuse * kAmbient;

	vec3 color = Lo + ambient;

	// Gamma Correction
	color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0/2.2));  

    PS_OUT_Color = vec4(color, 1.0);
}