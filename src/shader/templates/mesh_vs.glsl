#include <../common/mesh_vertex_attribs.glsl>
#include <../common/uniforms.glsl>

// ------------------------------------------------------------------
// OUTPUT VARIABLES  ------------------------------------------------
// ------------------------------------------------------------------

out vec3 PS_IN_CamPos;
out vec3 PS_IN_Position;
out vec4 PS_IN_NDCFragPos;
out vec4 PS_IN_ScreenPosition;
out vec4 PS_IN_LastScreenPosition;
out vec3 PS_IN_Normal;
out vec2 PS_IN_TexCoord;

#ifdef NORMAL_TEXTURE
	out vec3 PS_IN_Tangent;
	out vec3 PS_IN_Bitangent;
#endif

#ifdef DISPLACEMENT_TYPE_PARALLAX_OCCLUSION
	out vec3 PS_IN_TangentViewPos;
	out vec3 PS_IN_TangentFragPos;
#endif

// ------------------------------------------------------------------
// FUNCTIONS  -------------------------------------------------------
// ------------------------------------------------------------------

#ifndef VERTEX_SHADER_FUNC
#define VERTEX_SHADER_FUNC

void vertex_func(inout VertexProperties v)
{

}

#endif

// ------------------------------------------------------------------
// MAIN  ------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
	VertexProperties v;

	vec4 pos = modelMat * vec4(VS_IN_Position, 1.0f);
	v.PS_IN_Position = pos.xyz;

	// Calculate current and previous screen positions
	v.PS_IN_ScreenPosition = viewProj * pos;
	v.PS_IN_LastScreenPosition = lastViewProj * lastModelMat * vec4(VS_IN_Position, 1.0f);

	pos = viewProj * vec4(pos.xyz, 1.0f);
	
	mat3 normal_mat = mat3(modelMat);
	v.PS_IN_Normal = normal_mat * VS_IN_Normal;

#ifdef TEXTURE_NORMAL
	v.PS_IN_Tangent = normal_mat * VS_IN_Tangent;
	v.PS_IN_Bitangent = normal_mat * VS_IN_Bitangent;

	#ifdef DISPLACEMENT_TYPE_PARALLAX_OCCLUSION
		mat3 TBN = transpose(mat3(normalize(PS_IN_Tangent), normalize(-PS_IN_Bitangent), normalize(PS_IN_Normal)));
		v.PS_IN_TangentViewPos = TBN * viewPos.xyz;
		v.PS_IN_TangentFragPos = TBN * PS_IN_Position;
	#endif
#endif

	v.PS_IN_TexCoord = VS_IN_TexCoord;
	v.PS_IN_NDCFragPos = pos;

	vertex_func(v);

	// Set output vertex outputs
	PS_IN_Position = v.PS_IN_Position;
	PS_IN_NDCFragPos = v.PS_IN_NDCFragPos;
	PS_IN_ScreenPosition = v.PS_IN_ScreenPosition;
	PS_IN_LastScreenPosition = v.PS_IN_LastScreenPosition;
	PS_IN_Normal = v.PS_IN_Normal;
	PS_IN_TexCoord = v.PS_IN_TexCoord;

	#ifdef TEXTURE_NORMAL
		PS_IN_Tangent = v.PS_IN_Tangent;
		PS_IN_Bitangent = v.PS_IN_Bitangent;
	#endif

	#ifdef DISPLACEMENT_TYPE_PARALLAX_OCCLUSION
		PS_IN_TangentViewPos = v.PS_IN_TangentViewPos;
		PS_IN_TangentFragPos = v.PS_IN_TangentFragPos;
	#endif

	gl_Position = pos;
}

// ------------------------------------------------------------------