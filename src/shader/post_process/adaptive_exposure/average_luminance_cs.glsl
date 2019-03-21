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

layout (binding = 0, rgba32f) uniform image2D ;

uniform sampler2D u_Color;

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
	
	for(uint i = 0; i < width/(LUM_THREADS * AVG_LUM_THREADS); i++)
	{
		for(uint j = 0; j < height/16u; j++)
			total_luminance += imageLoad(i_Luma, ivec2(gl_GlobalInvocationID.x + AVG_LUM_THREADS * i, j));
	}

	avg_temp[gl_GlobalInvocationID.x] = total_luminance;
	
	groupMemoryBarrier();
	barrier();

	if (gl_GlobalInvocationID.x == 0)
	{
		for(uint i = 1; i < AVG_LUM_THREADS; i++)
			total_luminance += avg_temp[i];

		float luminance = total_luminance / ((width/AVG_LUM_THREADS)*(height/AVG_LUM_THREADS));
		imageStore(ivec2(0, 0), middleGrey/(exp(luminance)-DELTA));
	}
}

// ------------------------------------------------------------------