// ------------------------------------------------------------------
// INPUT VARIABLES  -------------------------------------------------
// ------------------------------------------------------------------

layout(location = 0) in vec3  VS_IN_Position;
layout(location = 1) in vec2  VS_IN_TexCoord;
layout(location = 2) in vec3  VS_IN_Normal;
layout(location = 3) in vec3  VS_IN_Tangent;
layout(location = 4) in vec4  VS_IN_Bitangent;

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

// ------------------------------------------------------------------
// OUTPUT VARIABLES  ------------------------------------------------
// ------------------------------------------------------------------

#define NORMAL_TEXTURE

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

// ------------------------------------------------------------------
// MAIN  ------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
	// Calculate world position
	vec4 pos = modelMat * vec4(VS_IN_Position, 1.0f);
	PS_IN_WorldPosition = pos.xyz;

	// Calculate current and previous screen positions
	PS_IN_ScreenPosition = viewProj * vec4(pos.xyz, 1.0f);
	PS_IN_LastScreenPosition = lastMvpMat * vec4(VS_IN_Position, 1.0f);
	
	// Calculate TBN vectors
	mat3 normal_mat = mat3(modelMat);
	PS_IN_Normal = normal_mat * VS_IN_Normal;

#ifdef NORMAL_TEXTURE
	PS_IN_Tangent = normal_mat * VS_IN_Tangent;
	PS_IN_Bitangent = normal_mat * VS_IN_Bitangent;
#endif

	// Camera position
	PS_IN_CamPos = viewPos.xyz;

	// Texture coordinates
	PS_IN_TexCoord = VS_IN_TexCoord;

	gl_Position = PS_IN_ScreenPosition;
}