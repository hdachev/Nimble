#include <../common/mesh_vertex_attribs.glsl>
#include <../common/uniforms.glsl>

// ------------------------------------------------------------------
// OUTPUT VARIABLES  ------------------------------------------------
// ------------------------------------------------------------------

out vec2 PS_IN_TexCoord;
out vec3 PS_IN_CamPos;
out vec3 PS_IN_WorldPosition;
out vec4 PS_IN_ScreenPosition;
out vec4 PS_IN_LastScreenPosition;
out vec3 PS_IN_Normal;

#ifdef NORMAL_TEXTURE
    out vec3 PS_IN_Tangent;
    out vec3 PS_IN_Bitangent;
#endif

#ifdef HEIGHT_TEXTURE
	out vec3 PS_IN_TangentViewPos;
	out vec3 PS_IN_TangentFragPos;
#endif

// ------------------------------------------------------------------
// MAIN  ------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
	// Calculate world position
	vec4 pos = modelMat * vec4(VS_IN_Position, 1.0f);
	PS_IN_WorldPosition = pos.xyz;

	// Calculate current and previous screen positions
	PS_IN_ScreenPosition = viewProj * pos;
	PS_IN_LastScreenPosition = lastMvpMat * vec4(VS_IN_Position, 1.0f);
	
	// Calculate TBN vectors
	mat3 normal_mat = mat3(modelMat);
	PS_IN_Normal = normal_mat * VS_IN_Normal;

#ifdef NORMAL_TEXTURE
	PS_IN_Tangent = normal_mat * VS_IN_Tangent;
	PS_IN_Bitangent = normal_mat * VS_IN_Bitangent;

#ifdef HEIGHT_TEXTURE
	mat3 TBN = mat3(normalize(PS_IN_Tangent), normalize(PS_IN_Bitangent), normalize(PS_IN_Normal));
	PS_IN_TangentFragPos = TBN * PS_IN_Position;
	PS_IN_TangentViewPos = TBN * viewPos.xyz;
#endif
#endif

	// Camera position
	PS_IN_CamPos = viewPos.xyz;

	// Texture coordinates
	PS_IN_TexCoord = VS_IN_TexCoord;

	gl_Position = PS_IN_ScreenPosition;
}

// ------------------------------------------------------------------