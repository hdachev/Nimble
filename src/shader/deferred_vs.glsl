layout (location = 0) in vec3 VS_IN_Position;
layout (location = 1) in vec2 VS_IN_TexCoord;

#define MAX_SHADOW_FRUSTUM 8

struct ShadowFrustum
{
	mat4  shadowMatrix;
	float farPlane;
};

layout (std140) uniform u_PerFrame
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
	float		  tanHalfFov;
	float		  aspectRatio;
	float		  nearPlane;
	float		  farPlane;
};

out vec2 PS_IN_TexCoord;
out vec2 PS_IN_ViewRay;

void main()
{
    PS_IN_TexCoord = VS_IN_TexCoord;

    PS_IN_ViewRay.x = VS_IN_Position.x * aspectRatio * tanHalfFov;
    PS_IN_ViewRay.y = VS_IN_Position.y * tanHalfFov;

    gl_Position = vec4(VS_IN_Position, 1.0);
}