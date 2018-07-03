#include <../common/uniforms.glsl>
#include <../common/helper.glsl>

// ------------------------------------------------------------------
// SAMPLERS  --------------------------------------------------------
// ------------------------------------------------------------------

#ifdef ALBEDO_TEXTURE
    uniform sampler2D s_Albedo;
#endif

#ifdef NORMAL_TEXTURE
    uniform sampler2D s_Normal;
#endif

#ifdef METALNESS_TEXTURE
    uniform sampler2D s_Metalness;
#endif

#ifdef ROUGHNESS_TEXTURE
    uniform sampler2D s_Roughness;
#endif

#ifdef HEIGHT_TEXTURE
    uniform sampler2D s_Displacement;
#endif

#ifdef EMISSIVE_TEXTURE
    uniform sampler2D s_Emissive;
#endif

// ------------------------------------------------------------------
// INPUT VARIABLES  -------------------------------------------------
// ------------------------------------------------------------------

in vec2 PS_IN_TexCoord;
in vec4 PS_IN_ScreenPosition;
in vec4 PS_IN_LastScreenPosition;
in vec3 PS_IN_Normal;

#ifdef NORMAL_TEXTURE
    in vec3 PS_IN_Tangent;
    in vec3 PS_IN_Bitangent;
#endif

#ifdef HEIGHT_TEXTURE
	in vec3 PS_IN_TangentViewPos;
	in vec3 PS_IN_TangentFragPos;
#endif

// ------------------------------------------------------------------
// OUTPUT VARIABLES  ------------------------------------------------
// ------------------------------------------------------------------

layout (location = 0) out vec4 PS_OUT_Albedo;
layout (location = 1) out vec4 PS_OUT_Motion;
layout (location = 2) out vec3 PS_OUT_Normal;
layout (location = 3) out vec4 PS_OUT_MetalRoughEmissive;

// ------------------------------------------------------------------
// FUNCTIONS --------------------------------------------------------
// ------------------------------------------------------------------

vec2 tex_coord()
{
#ifdef HEIGHT_TEXTURE
    return parallax_occlusion_tex_coords(normalize(PS_IN_TangentViewPos - PS_IN_TangentFragPos), PS_IN_TexCoord, 0.05, s_Displacement); 
#else
    return PS_IN_TexCoord;
#endif
}

vec4 albedo(vec2 tex_coord)
{
#ifdef ALBEDO_TEXTURE
    return texture(s_Albedo, tex_coord);
#else
    return albedo_color;
#endif
}

vec3 normal(vec2 tex_coord)
{
#ifdef NORMAL_TEXTURE
    return get_normal_from_map(PS_IN_Tangent, PS_IN_Bitangent, PS_IN_Normal, tex_coord, s_Normal);
#else
    return PS_IN_Normal;
#endif
}

float metalness(vec2 tex_coord)
{
#ifdef METALNESS_TEXTURE
    return texture(s_Metalness, tex_coord).r;
#else
    return metalness_roughness.r;
#endif
}

float roughness(vec2 tex_coord)
{
#ifdef ROUGHNESS_TEXTURE
    return texture(s_Roughness, tex_coord).r;
#else
    return metalness_roughness.g;
#endif
}

float emissive(vec2 tex_coord)
{
#ifdef EMISSIVE_TEXTURE
    return texture(s_Emissive, tex_coord).rgb;
#else
    return 0.0;
#endif
}

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
    // Get texture coordinate
    vec2 tex_coord = tex_coord();

    // Store albedo color
    vec4 albedo_color = albedo(tex_coord);
    
    if (albedo_color.a < 0.5)
        discard;
        
    PS_OUT_Albedo = albedo_color;

    // Store encoded normal vector
    //vec3 normal = normal(tex_coord);
    //vec2 encoded_normal = encode_normal(normal);
    PS_OUT_Normal = normal(tex_coord);;

    // Store motion vector
    vec2 motion = motion_vector(PS_IN_LastScreenPosition, PS_IN_ScreenPosition);

    PS_OUT_Motion = vec4(0.0, 0.0, motion.x, motion.y);

    // Store metalness
    PS_OUT_MetalRoughEmissive.r = metalness(tex_coord);

    // Store roughness
    PS_OUT_MetalRoughEmissive.g = roughness(tex_coord);

    // Store emissive color
    PS_OUT_MetalRoughEmissive.b = emissive(tex_coord);
}

// ------------------------------------------------------------------