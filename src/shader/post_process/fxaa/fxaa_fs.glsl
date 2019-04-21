#include <fxaa.glsl>

// ------------------------------------------------------------------
// INPUTS -----------------------------------------------------------
// ------------------------------------------------------------------

in vec2 FS_IN_TexCoord;

// ------------------------------------------------------------------
// OUTPUTS ----------------------------------------------------------
// ------------------------------------------------------------------

out vec4 FS_OUT_Color;

// ------------------------------------------------------------------
// UNIFORMS ---------------------------------------------------------
// ------------------------------------------------------------------

uniform sampler2D s_Texture;

uniform float u_QualityEdgeThreshold;
uniform float u_QualityEdgeThresholdMin;
uniform vec2 u_QualityRcpFrame;

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
	FS_OUT_Color = FxaaPixelShader(FS_IN_TexCoord, 
								   vec4(0.0),
								   s_Texture,
								   s_Texture,
								   s_Texture,
								   u_QualityRcpFrame,
								   vec4(0.0),
								   vec4(0.0),
								   vec4(0.0),
								   0.75,
								   u_QualityEdgeThreshold,
								   u_QualityEdgeThresholdMin,
								   0.0,
								   0.0,
								   0.0,
								   vec4(0.0));


}

// ------------------------------------------------------------------