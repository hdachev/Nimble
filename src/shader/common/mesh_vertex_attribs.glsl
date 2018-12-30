// ------------------------------------------------------------------
// INPUT VARIABLES  -------------------------------------------------
// ------------------------------------------------------------------

#ifdef VERTEX_COLORS
    layout(location = 0) in vec3  VS_IN_Position;
    layout(location = 1) in vec2  VS_IN_TexCoord;
    layout(location = 2) in vec3  VS_IN_Normal;
    layout(location = 3) in vec3  VS_IN_Tangent;
    layout(location = 4) in vec3  VS_IN_Bitangent;
    layout(location = 5) in vec4  VS_IN_Color;
    #ifdef MESH_TYPE_SKELETAL
        layout(location = 6) in vec4  VS_IN_BoneWeights;
        layout(location = 7) in ivec4  VS_IN_BoneIndices;
    #endif
#else
    layout(location = 0) in vec3  VS_IN_Position;
    layout(location = 1) in vec2  VS_IN_TexCoord;
    layout(location = 2) in vec3  VS_IN_Normal;
    layout(location = 3) in vec3  VS_IN_Tangent;
    layout(location = 4) in vec3  VS_IN_Bitangent;
    #ifdef MESH_TYPE_SKELETAL
        layout(location = 5) in vec4  VS_IN_BoneWeights;
        layout(location = 6) in ivec4  VS_IN_BoneIndices;
    #endif
#endif

// ------------------------------------------------------------------