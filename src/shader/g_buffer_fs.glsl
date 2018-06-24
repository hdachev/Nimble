// ------------------------------------------------------------------
// UNIFORM BUFFERS --------------------------------------------------
// ------------------------------------------------------------------

layout (std140) uniform u_PerMaterial //#binding 2
{ 
    vec4 albedo_color;
    vec4 metalness_roughness;
};

// ------------------------------------------------------------------
// SAMPLERS  --------------------------------------------------------
// ------------------------------------------------------------------

#define ALBEDO_TEXTURE

#ifdef ALBEDO_TEXTURE
    uniform sampler2D s_Albedo;
#endif

#define NORMAL_TEXTURE

#ifdef NORMAL_TEXTURE
    uniform sampler2D s_Normal;
#endif

#define METALNESS_TEXTURE

#ifdef METALNESS_TEXTURE
    uniform sampler2D s_Metalness;
#endif

#define ROUGHNESS_TEXTURE

#ifdef ROUGHNESS_TEXTURE
    uniform sampler2D s_Roughness;
#endif

#ifdef HEIGHT_TEXTURE
    uniform sampler2D s_Height;
#endif

#ifdef EMISSIVE_TEXTURE
    uniform sampler2D s_Emissive;
#endif

// ------------------------------------------------------------------
// INPUT VARIABLES  -------------------------------------------------
// ------------------------------------------------------------------

in vec2 PS_IN_TexCoord;
in vec3 PS_IN_CamPos;
in vec3 PS_IN_WorldPosition;
in vec4 PS_IN_ScreenPosition;
in vec4 PS_IN_LastScreenPosition;
in vec3 PS_IN_Normal;

#ifdef NORMAL_TEXTURE
    in vec3 PS_IN_Tangent;
    in vec3 PS_IN_Bitangent;
#endif

// ------------------------------------------------------------------
// OUTPUT VARIABLES  ------------------------------------------------
// ------------------------------------------------------------------

layout (location = 0) out vec4 PS_OUT_Albedo;
layout (location = 1) out vec2 PS_OUT_NormalMotion;
layout (location = 2) out vec2 PS_OUT_MetalRoughEmissive;

// ------------------------------------------------------------------
// FUNCTIONS --------------------------------------------------------
// ------------------------------------------------------------------

vec2 encode_normal(vec3 n)
{
    half f = sqrt(8*n.z+8);
    return n.xy / f + 0.5;
}

vec2 tex_coord()
{
#ifdef HEIGHT_TEXTURE
    return vec2();
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
    // Create TBN matrix.
    mat3 TBN = mat3(normalize(PS_IN_Tangent), normalize(PS_IN_Bitangent), normalize(PS_IN_Normal));

    // Sample tangent space normal vector from normal map and remap it from [0, 1] to [-1, 1] range.
    vec3 n = normalize(texture(s_Normal, tex_coord).xyz * 2.0 - 1.0);

    // Multiple vector by the TBN matrix to transform the normal from tangent space to world space.
    n = normalize(TBN * n);

    return n;
#else
    return PS_IN_Normal;
#endif
}

float metalness(vec2 tex_coord)
{
#ifdef METALNESS_TEXTURE
    return texture(s_Roughness, tex_coord).r;
#else
    return metalness_roughness.r;
#endif
}

float roughness(vec2 tex_coord)
{
#ifdef ROUGHNESS_TEXTURE
    return texture(s_Metalness, tex_coord).r;
#else
    return metalness_roughness.g;
#endif
}

vec3 emissive(vec2 tex_coord)
{
#ifdef EMISSIVE_TEXTURE
    return texture(s_Emissive, tex_coord).rgb;
#else
    return vec3(0.0);
#endif
}

vec2 motion_vector()
{
    // Perspective division and remapping to [0, 1] range.
    vec2 current = (PS_IN_ScreenPosition.xy / PS_IN_ScreenPosition.w) * 0.5 + 0.5;
    vec2 last = (PS_IN_LastScreenPosition.xy / PS_IN_LastScreenPosition.w) * 0.5 + 0.5;

    return current - last;
}

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
    // Get texture coordinate
    vec2 tex_coord = tex_coord();

    // Store albedo color
    PS_OUT_Albedo = albedo(tex_coord);

    // Store encoded normal vector
    vec3 normal = normal(tex_coord);
    PS_OUT_Normal.rg = encode_normal(normal);

    // Store motion vector
    PS_OUT_NormalMotion.ba = motion_vector();

    // Store metalness
    PS_OUT_MetalRough.r = metalness(tex_coord);

    // Store roughness
    PS_OUT_MetalRough.g = roughness(tex_coord);

    // Store emissive color
    PS_OUT_Emissive.rgb = emissive(tex_coord);
}