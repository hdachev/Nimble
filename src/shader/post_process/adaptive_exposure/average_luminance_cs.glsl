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
// UNIFORMS ---------------------------------------------------------
// ------------------------------------------------------------------

layout (binding = 0, rgba32f) uniform image2D;

uniform int u_Width;
uniform int u_Height;
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
	float total_luminance = 0.0f;
	
	for(uint i = 0; i < u_Width/(LUM_THREADS * AVG_LUM_THREADS); i++)
	{
		for(uint j = 0; j < u_Height/16; j++)
			total_luminance += imageLoad(i_Luma, ivec2(gl_GlobalInvocationID.x + AVG_LUM_THREADS * i, j));
	}

	avg_temp[gl_GlobalInvocationID.x] = total_luminance;
	
	groupMemoryBarrier();
	barrier();

	if (gl_GlobalInvocationID.x == 0)
	{
		for(uint i = 1; i < AVG_LUM_THREADS; i++)
			total_luminance += avg_temp[i];

		float luminance = total_luminance / ((u_Width / AVG_LUM_THREADS) * (u_Height / AVG_LUM_THREADS));
		imageStore(ivec2(0, 0), u_MiddleGrey / (exp(luminance) - DELTA));
	}
}

// ------------------------------------------------------------------