out vec4 FragColor;

in vec2 PS_IN_TexCoord;

#define SHOW_COLOR 0
#define SHOW_DEPTH 1
#define SHOW_NORMALS 2
#define SHOW_ROUGHNESS 3
#define SHOW_METALNESS 4
#define SHOW_MOTION_VECTORS 5
#define SHOW_EMISSIVE 6
#define SHOW_DISPLACEMENT 7
#define SHOW_SHADOW_MAPS 8

uniform int u_CurrentOutput;
uniform float u_FarPlane;
uniform float u_NearPlane;

uniform sampler2D s_Color;
uniform sampler2D s_Depth;
uniform sampler2DArray s_CSMShadowMaps;

float GetLinearDepth()
{
    float f = u_FarPlane;
    float n = u_NearPlane;
    float z = (2 * n) / (f + n - texture( s_Depth, PS_IN_TexCoord ).x * (f - n));
    return z;
}

void main()
{
	if (u_CurrentOutput == SHOW_COLOR)
		FragColor = vec4(texture(s_Color, PS_IN_TexCoord).xyz, 1.0);
	else if (u_CurrentOutput == SHOW_DEPTH)
		FragColor = vec4(vec3(GetLinearDepth()), 1.0);
	else if (u_CurrentOutput >= SHOW_SHADOW_MAPS)
		FragColor = vec4(vec3(texture(s_CSMShadowMaps, vec3(PS_IN_TexCoord, float(u_CurrentOutput - SHOW_SHADOW_MAPS))).x), 1.0);
	else 
		FragColor = vec4(1.0);
}