#include <../../common/uniforms.glsl>
#include <../../common/helper.glsl>

// Average Luminance Compute Shader based off:
// https://software.intel.com/en-us/articles/hdr-rendering-with-compute-shader-sample

// ------------------------------------------------------------------
// DEFINES ----------------------------------------------------------
// ------------------------------------------------------------------

#define LUM_THREADS 8
#define AVG_LUM_THREADS 8
#define DELTA 0.00000001

// ------------------------------------------------------------------
// INPUTS -----------------------------------------------------------
// ------------------------------------------------------------------

layout (local_size_x = AVG_LUM_THREADS, local_size_y = 1) in;

// ------------------------------------------------------------------
// CONSTANTS --------------------------------------------------------
// ------------------------------------------------------------------

const int kSize = 512;

// ------------------------------------------------------------------
// UNIFORMS ---------------------------------------------------------
// ------------------------------------------------------------------

layout (binding = 0, r32f) uniform image2D i_Luma;
layout (binding = 1, r32f) uniform image2D i_AvgLuma;

uniform float u_MiddleGrey;

// ------------------------------------------------------------------
// GLOBALS ----------------------------------------------------------
// ------------------------------------------------------------------

shared float avg_temp[AVG_LUM_THREADS];

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
	float total_luminance = 0.0;
	
	for(uint i = 0; i < kSize/(LUM_THREADS * AVG_LUM_THREADS); i++)
	{
		for(uint j = 0; j < kSize/16; j++)
			total_luminance += imageLoad(i_Luma, ivec2(gl_GlobalInvocationID.x + AVG_LUM_THREADS * i, j)).x;
	}

	avg_temp[gl_GlobalInvocationID.x] = total_luminance;
	
	groupMemoryBarrier();
	barrier();

	if (gl_GlobalInvocationID.x == 0)
	{
		for(uint i = 1; i < AVG_LUM_THREADS; i++)
			total_luminance += avg_temp[i];

		float luminance = total_luminance / ((kSize / AVG_LUM_THREADS) * (kSize / AVG_LUM_THREADS));
		imageStore(i_AvgLuma, ivec2(0, 0), vec4(u_MiddleGrey / (exp(luminance) - DELTA), 0, 0, 0));
	}
}

// ------------------------------------------------------------------