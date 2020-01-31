// ------------------------------------------------------------------
// DEFINITIONS ------------------------------------------------------
// ------------------------------------------------------------------

#define MAX_SHADOW_MAP_CASCADES 4
#define MAX_RELFECTION_PROBES 128
#define MAX_GI_PROBES 128
#define MAX_POINT_LIGHTS 10000
#define MAX_SPOT_LIGHTS 10000
#define MAX_DIRECTIONAL_LIGHTS 512
#define MAX_SHADOW_CASTING_POINT_LIGHTS 8
#define MAX_SHADOW_CASTING_SPOT_LIGHTS 8
#define MAX_SHADOW_CASTING_DIRECTIONAL_LIGHTS 8
#define MAX_BONES 100
#define MAX_LIGHTS 10000
#define LIGHT_TYPE_DIRECTIONAL 0
#define LIGHT_TYPE_SPOT 1
#define LIGHT_TYPE_POINT 2

// ------------------------------------------------------------------
// STRUCTURES -------------------------------------------------------
// ------------------------------------------------------------------

struct LightData
{
                    // | Spot                     | Directional                                        | Point                    |
    ivec4 indices0; // | y: shadow map, z: matrix | y: first shadow map index, z: first cascade matrix | y: shadow map            | 
    ivec4 indices1; // |                          |                                                    |                          |
    vec4 data0;     // | xyz: position, w: bias   | xyz: direction, w: bias                            | xyz: positon, w: bias    |
    vec4 data1;     // | xy: cutoff               | xyz: color, w: intensity                           | x: near, y: far          |
    vec4 data2;     // | xyz: direction, w: range | xyzw: far planes                                   | xyz: color, w: intensity |
    vec4 data3;     // | xyz: color, w: intensity |                                                    |                          |
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
	vec2 TexCoords;
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
	float NdotV;
};

#ifdef DIRECTIONAL_LIGHT_SHADOW_MAPPING
	float directional_light_shadows(in FragmentProperties f, int shadow_map_idx, int light_idx);
	vec3 csm_debug_color(float frag_depth, int shadow_map_idx);
#endif

#ifdef SPOT_LIGHT_SHADOW_MAPPING
	float spot_light_shadows(in FragmentProperties f, int shadow_map_idx, int light_idx);
#endif

#ifdef POINT_LIGHT_SHADOW_MAPPING
	float point_light_shadows(in FragmentProperties f, int shadow_map_idx, int light_idx);
#endif

// ------------------------------------------------------------------