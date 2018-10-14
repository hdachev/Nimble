#include <../../common/uniforms.glsl>
#include <../../common/helper.glsl>

out vec2 PS_OUT_FragColor;

in vec2 PS_IN_TexCoord;

uniform float u_NearBegin;
uniform float u_NearEnd;
uniform float u_FarBegin;
uniform float u_FarEnd;

uniform sampler2D s_Depth;

// ----------------------------------------------------------------------
// Generates Near and Far CoC values using View Space Depth.
// ----------------------------------------------------------------------

void main()
{
	float depth_ndc = texture(s_Depth, PS_IN_TexCoord).x;
	float depth = get_view_space_depth(PS_IN_TexCoord, depth_ndc);
	
	// Calculate Near CoC
	float nearCOC = 0.0;

	if (depth < u_NearEnd)
		nearCOC = 1.0 / (u_NearBegin - u_NearEnd) * depth + -u_NearEnd / (u_NearBegin - u_NearEnd);
	else if (depth < u_NearBegin)
		nearCOC = 1.0;

	nearCOC = clamp(nearCOC, 0.0, 1.0);
	
	// Calculate Far CoC
	float farCOC = 1.0;

	if (depth < u_FarBegin)
		farCOC = 0.0;
	else if (depth < u_FarEnd)
		farCOC = 1.0 / (u_FarEnd - u_FarBegin) * depth + -u_FarBegin / (u_FarEnd - u_FarBegin);

	farCOC = clamp(farCOC, 0.0, 1.0);
 
	PS_OUT_FragColor = vec2(nearCOC, farCOC);
}