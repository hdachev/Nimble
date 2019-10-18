#include <../common/mesh_vertex_attribs.glsl>
#include <../common/uniforms.glsl>

// ------------------------------------------------------------------
// OUTPUT VARIABLES  ------------------------------------------------
// ------------------------------------------------------------------

out vec3 FS_IN_Position;
out vec4 FS_IN_NDCFragPos;
out vec4 FS_IN_ScreenPosition;
out vec4 FS_IN_LastScreenPosition;
out vec3 FS_IN_Normal;
out vec2 FS_IN_TexCoord;

#ifdef TEXTURE_NORMAL
	out vec3 FS_IN_Tangent;
	out vec3 FS_IN_Bitangent;
#endif

#ifdef DISPLACEMENT_TYPE_PARALLAX_OCCLUSION
	out vec3 FS_IN_TangentViewPos;
	out vec3 FS_IN_TangentFragPos;
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

	vec4 pos = model_mat * vec4(VS_IN_Position, 1.0f);
	v.Position = pos.xyz;

	// Calculate current and previous screen positions
	v.ScreenPosition = view_proj * pos;
	v.LastScreenPosition = last_view_proj * last_model_mat * vec4(VS_IN_Position, 1.0f);

	pos = view_proj * vec4(pos.xyz, 1.0f);
	
	mat3 normal_mat = mat3(model_mat);
	v.Normal = normal_mat * VS_IN_Normal;

#ifdef TEXTURE_NORMAL
	v.Tangent = normal_mat * VS_IN_Tangent;
	v.Bitangent = normal_mat * VS_IN_Bitangent;

	#ifdef DISPLACEMENT_TYPE_PARALLAX_OCCLUSION
		mat3 TBN = transpose(mat3(normalize(FS_IN_Tangent), normalize(-FS_IN_Bitangent), normalize(FS_IN_Normal)));
		v.TangentViewPos = TBN * view_pos.xyz;
		v.TangentFragPos = TBN * FS_IN_Position;
	#endif
#endif

	v.TexCoord = VS_IN_TexCoord;
	v.NDCFragPos = pos;

	vertex_func(v);

	// Set output vertex outputs
	FS_IN_Position = v.Position;
	FS_IN_NDCFragPos = v.NDCFragPos;
	FS_IN_ScreenPosition = v.ScreenPosition;
	FS_IN_LastScreenPosition = v.LastScreenPosition;
	FS_IN_Normal = v.Normal;
	FS_IN_TexCoord = v.TexCoord;

	#ifdef TEXTURE_NORMAL
		FS_IN_Tangent = v.Tangent;
		FS_IN_Bitangent = v.Bitangent;
	#endif

	#ifdef DISPLACEMENT_TYPE_PARALLAX_OCCLUSION
		FS_IN_TangentViewPos = v.TangentViewPos;
		FS_IN_TangentFragPos = v.TangentFragPos;
	#endif

	gl_Position = pos;
}

// ------------------------------------------------------------------