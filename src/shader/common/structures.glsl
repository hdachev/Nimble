// ------------------------------------------------------------------
// DEFINITIONS ------------------------------------------------------
// ------------------------------------------------------------------

#define MAX_SHADOW_FRUSTUM 8
#define MAX_POINT_LIGHTS 32
#define MAX_BONES 100

// ------------------------------------------------------------------
// STRUCTURES -------------------------------------------------------
// ------------------------------------------------------------------

struct ShadowFrustum
{
	mat4  shadowMatrix;
	float farPlane;
};

// ------------------------------------------------------------------

struct PointLight
{
	vec4 position;
	vec4 color;
};

// ------------------------------------------------------------------

struct DirectionalLight
{
	vec4 direction;
	vec4 color;
};

// ------------------------------------------------------------------

struct VertexProperties
{
	vec3 Position;
	vec4 NDCFragPos;
	vec4 ScreenPosition;
	vec4 LastScreenPosition;
	vec3 Normal;
	vec2 TexCoord;

#ifdef TEXTURE_NORMAL
	vec3 Tangent;
	vec3 Bitangent;
#endif

#ifdef DISPLACEMENT_TYPE_PARALLAX_OCCLUSION
	vec3 TangentViewPos;
	vec3 TangentFragPos;
#endif
};

// ------------------------------------------------------------------

struct FragmentProperties
{
	vec3 Position;
	vec3 Normal;
	float FragDepth;
#ifdef TEXTURE_NORMAL
	vec3 Tangent;
	vec3 Bitangent;
#endif
#ifdef DISPLACEMENT_TYPE_PARALLAX_OCCLUSION
	vec3 TangentViewPos;
	vec3 TangentFragPos;
#endif
};

// ------------------------------------------------------------------

struct MaterialProperties
{
	vec4 albedo;
	vec3 normal;
	float metallic;
	float roughness;
};

// ------------------------------------------------------------------

struct PBRProperties
{
	vec3 N;
	vec3 V;
	vec3 R;
	vec3 F0;
	vec3 F;
	vec3 kS;
	vec3 kD;
	float NdotV
};

// ------------------------------------------------------------------