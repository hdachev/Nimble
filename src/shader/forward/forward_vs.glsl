#include <../common/mesh_vertex_attribs.glsl>
#include <../common/uniforms.glsl>

// ------------------------------------------------------------------
// OUTPUT VARIABLES  ------------------------------------------------
// ------------------------------------------------------------------

out vec3 PS_IN_CamPos;
out vec3 PS_IN_Position;
out vec4 PS_IN_NDCFragPos;
out vec3 PS_IN_Normal;
out vec2 PS_IN_TexCoord;

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
	vec4 pos = modelMat * vec4(VS_IN_Position, 1.0f);
	PS_IN_Position = pos.xyz;

	pos = projMat * viewMat * vec4(pos.xyz, 1.0f);
	
	mat3 normal_mat = mat3(modelMat);
	PS_IN_Normal = normal_mat * VS_IN_Normal;

#ifdef NORMAL_TEXTURE
	PS_IN_Tangent = normal_mat * VS_IN_Tangent;
	PS_IN_Bitangent = normal_mat * VS_IN_Bitangent;

#ifdef HEIGHT_TEXTURE
	mat3 TBN = transpose(mat3(normalize(PS_IN_Tangent), normalize(-PS_IN_Bitangent), normalize(PS_IN_Normal)));
	PS_IN_TangentViewPos = TBN * viewPos.xyz;
	PS_IN_TangentFragPos = TBN * PS_IN_Position;
#endif
#endif

	PS_IN_CamPos = viewPos.xyz;
	PS_IN_TexCoord = VS_IN_TexCoord;

	PS_IN_NDCFragPos = pos;
	gl_Position = pos;
}

// ------------------------------------------------------------------