#include <../../common/uniforms.glsl>
#include <../../common/helper.glsl>

// Average Luminance Compute Shader based off:
// https://software.intel.com/en-us/articles/hdr-rendering-with-compute-shader-sample

// ------------------------------------------------------------------
// DEFINES ----------------------------------------------------------
// ------------------------------------------------------------------

#define LUM_THREADS 8
#define DELTA 0.00000001

// ------------------------------------------------------------------
// INPUTS -----------------------------------------------------------
// ------------------------------------------------------------------

layout (local_size_x = LUM_THREADS, local_size_y = LUM_THREADS) in;

// ------------------------------------------------------------------
// UNIFORMS ---------------------------------------------------------
// ------------------------------------------------------------------

layout (binding = 0, rgba32f) uniform image2D i_Luma;

uniform sampler2D u_Color;

// ------------------------------------------------------------------
// GLOBALS ----------------------------------------------------------
// ------------------------------------------------------------------

shared float temp0[LUM_THREADS][LUM_THREADS];
shared float temp1[LUM_THREADS][LUM_THREADS];

// ------------------------------------------------------------------
// FUNCTIONS --------------------------------------------------------
// ------------------------------------------------------------------

float log_luminance(vec2 tex_coord, ivec2 offset)
{
	vec3 color = textureOffset(u_Color, tex_coord, offset).rgb;
	float luma = luminance(color);

	return log(luma + DELTA);
}

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
	vec2 tex_coord = (2 * gl_GlobalInvocationID.xy + 0.5) / vec2(width, height);

	float avg = 0;

	ivec2 offset = ivec2(0, 0);
	avg += log_luminance(tex_coord, offset * 2);

	offset = ivec2(0, 1);
	avg += log_luminance(tex_coord, offset * 2);
	
	offset = ivec2(1, 0);
	avg += log_luminance(tex_coord, offset * 2);
	
	offset = ivec2(1, 1);
	avg += log_luminance(tex_coord, offset * 2);
	
	avg = avg / 4;
	temp0[gl_LocalInvocationID.x][gl_LocalInvocationID.y] = avg;

	groupMemoryBarrier();
	barrier();

	if (gl_LocalInvocationID.x < LUM_AND_BRIGHT_THREADS / 2 && gl_LocalInvocationID.y < LUM_AND_BRIGHT_THREADS / 2) 
	{
		float nextLevel;
		nextLevel = temp0[gl_LocalInvocationID.x * 2][gl_LocalInvocationID.y * 2];
		nextLevel += temp0[gl_LocalInvocationID.x * 2 + 1][gl_LocalInvocationID.y * 2];
		nextLevel += temp0[gl_LocalInvocationID.x * 2][gl_LocalInvocationID.y * 2 + 1];
		nextLevel += temp0[gl_LocalInvocationID.x * 2 + 1][gl_LocalInvocationID.y * 2 + 1];
		nextLevel = nextLevel / 4;
		temp1[gl_LocalInvocationID.x][gl_LocalInvocationID.y] = nextLevel;
	}

	groupMemoryBarrier();	
	barrier();

	if (gl_LocalInvocationID.x < LUM_AND_BRIGHT_THREADS / 4 && gl_LocalInvocationID.y < LUM_AND_BRIGHT_THREADS / 4)
	{
		float nextLevel;
		nextLevel =  temp1[gl_LocalInvocationID.x * 2][gl_LocalInvocationID.y * 2];
		nextLevel += temp1[gl_LocalInvocationID.x * 2 + 1][gl_LocalInvocationID.y * 2];
		nextLevel += temp1[gl_LocalInvocationID.x * 2][gl_LocalInvocationID.y * 2 + 1];
		nextLevel += temp1[gl_LocalInvocationID.x * 2 + 1][gl_LocalInvocationID.y * 2 + 1];
		nextLevel = nextLevel / 4;
		temp0[gl_LocalInvocationID.x][gl_LocalInvocationID.y] = nextLevel;
	}

	groupMemoryBarrier();
	barrier();

	if (gl_LocalInvocationID.x == 0 && gl_LocalInvocationID.y == 0)
	{
		float nextLevel;
		nextLevel =  temp0[0][0];
		nextLevel += temp0[1][0];
		nextLevel += temp0[0][1];
		nextLevel += temp0[1][1];
		nextLevel = nextLevel / 4;
		imageStore(i_Luma, gl_WorkGroupID.xy, nextLevel);
	}
}

// ------------------------------------------------------------------