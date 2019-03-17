#include <../../common/uniforms.glsl>
#include <../../common/helper.glsl>

#define UPSAMPLE_DEPTH_THRESHOLD 1.5f

// ------------------------------------------------------------------
// OUTPUTS ----------------------------------------------------------
// ------------------------------------------------------------------

out vec3 FS_OUT_FragColor;

// ------------------------------------------------------------------
// INPUTS -----------------------------------------------------------
// ------------------------------------------------------------------

in vec2 FS_IN_TexCoord;

// ------------------------------------------------------------------
// UNIFORMS ---------------------------------------------------------
// ------------------------------------------------------------------

uniform sampler2D s_Depth;
uniform sampler2D s_Volumetric;

// ------------------------------------------------------------------
// FUNCTIONS --------------------------------------------------------
// ------------------------------------------------------------------

vec3 bilateral_upsample(vec2 tex_coord)
{
    const float threshold = UPSAMPLE_DEPTH_THRESHOLD;
    vec4 highResDepth = vec4(depth_exp_to_view(near_plane, far_plane, textureLod(s_Depth, tex_coord, 0.0).r));
	
	vec2 uv00 = tex_coord - 0.5 * u_PixelSize;
	vec2 uv10 = uv00 + vec2(u_PixelSize.x, 0.0);
	vec2 uv01 = uv00 + vec2(0.0, u_PixelSize.y);
	vec2 uv11 = uv00 + u_PixelSize;

	vec4 lowResDepth;

    lowResDepth[0] = depth_exp_to_view(near_plane, far_plane, textureLod(s_Depth, uv00, 1.0).r);
    lowResDepth[1] = depth_exp_to_view(near_plane, far_plane, textureLod(s_Depth, uv10, 1.0).r);
    lowResDepth[2] = depth_exp_to_view(near_plane, far_plane, textureLod(s_Depth, uv01, 1.0).r);
    lowResDepth[3] = depth_exp_to_view(near_plane, far_plane, textureLod(s_Depth, uv11, 1.0).r);

	vec4 depthDiff = abs(lowResDepth - highResDepth);
	float accumDiff = dot(depthDiff, vec4(1, 1, 1, 1));
	
	if (accumDiff < threshold) // small error, not an edge -> use bilinear filter
		return texture(s_Volumetric, tex_coord).rgb;
    
	// find nearest sample
	float minDepthDiff = depthDiff[0];
	vec2 nearestUv = uv00;

	if (depthDiff[1] < minDepthDiff)
	{
		nearestUv = uv10;
		minDepthDiff = depthDiff[1];
	}
	if (depthDiff[2] < minDepthDiff)
	{
		nearestUv = uv01;
		minDepthDiff = depthDiff[2];
	}
	if (depthDiff[3] < minDepthDiff)
	{
		nearestUv = uv11;
		minDepthDiff = depthDiff[3];
	}

    return texture(s_Volumetric, nearestUv).rgb;
}

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
	FS_OUT_FragColor = bilateral_upsample(FS_IN_TexCoord);
}

// ------------------------------------------------------------------