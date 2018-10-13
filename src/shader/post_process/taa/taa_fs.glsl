#include <../../common/uniforms.glsl>

// ------------------------------------------------------------------
// OUTPUTS ----------------------------------------------------------
// ------------------------------------------------------------------

out vec3 PS_OUT_FragColor;

// ------------------------------------------------------------------
// INPUTS -----------------------------------------------------------
// ------------------------------------------------------------------

in vec2 PS_IN_TexCoord;

// ------------------------------------------------------------------
// UNIFORMS ---------------------------------------------------------
// ------------------------------------------------------------------

uniform sampler2D s_Color;
uniform sampler2D s_History;
uniform sampler2D s_Velocity;
uniform vec2 u_PixelSize;
uniform int u_FirstFrame;

#define HDR_COLOR_BUFFER
#define UNJITTER_TEX_COORDS

// ------------------------------------------------------------------
// CONSTANTS --------------------------------------------------------
// ------------------------------------------------------------------

const float FLT_EPS = 0.00000001f;

// ------------------------------------------------------------------
// FUNCTIONS --------------------------------------------------------
// ------------------------------------------------------------------

vec3 clip_aabb(vec3 aabb_min, vec3 aabb_max, vec3 p, vec3 q)
{
#ifdef USE_OPTIMIZATIONS
	// note: only clips towards aabb center (but fast!)
	vec3 p_clip = 0.5 * (aabb_max + aabb_min);
	vec3 e_clip = 0.5 * (aabb_max - aabb_min) + FLT_EPS;
	vec3 v_clip = q - p_clip;
	vec3 v_unit = v_clip / e_clip;
	vec3 a_unit = abs(v_unit);
	float ma_unit = max(a_unit.x, max(a_unit.y, a_unit.z));
	if (ma_unit > 1.0)
		return p_clip + v_clip / ma_unit;
	else
		return q;// point inside aabb
#else
	vec3 r = q - p;
	vec3 rmax = aabb_max - p.xyz;
	vec3 rmin = aabb_min - p.xyz;
	const float eps = FLT_EPS;
	if (r.x > rmax.x + eps)
		r *= (rmax.x / r.x);
	if (r.y > rmax.y + eps)
		r *= (rmax.y / r.y);
	if (r.z > rmax.z + eps)
		r *= (rmax.z / r.z);
	if (r.x < rmin.x - eps)
		r *= (rmin.x / r.x);
	if (r.y < rmin.y - eps)
		r *= (rmin.y / r.y);
	if (r.z < rmin.z - eps)
		r *= (rmin.z / r.z);
	return p + r;
#endif
}

// ------------------------------------------------------------------

// https://software.intel.com/en-us/node/503873
vec3 RGB_YCoCg(vec3 c)
{
	// Y = R/4 + G/2 + B/4
	// Co = R/2 - B/2
	// Cg = -R/4 + G/2 - B/4
	return vec3(
		 c.x/4.0 + c.y/2.0 + c.z/4.0,
		 c.x/2.0 - c.z/2.0,
		-c.x/4.0 + c.y/2.0 - c.z/4.0
	);
}

// ------------------------------------------------------------------

// https://software.intel.com/en-us/node/503873
vec3 YCoCg_RGB(vec3 c)
{
	// R = Y + Co - Cg
	// G = Y + Cg
	// B = Y - Co - Cg
	return clamp(vec3(
		c.x + c.y - c.z,
		c.x + c.z,
		c.x - c.y - c.z), 
        vec3(0.0), 
        vec3(1.0));
}

// ------------------------------------------------------------------

vec4 sample_color(sampler2D tex, vec2 uv)
{
#ifdef USE_YCOCG
	vec4 c = texture(tex, uv);
	return vec4(RGB_YCoCg(c.rgb), c.a);
#else
	return texture(tex, uv);
#endif
}

// ------------------------------------------------------------------

vec4 resolve_color(vec4 c)
{
#ifdef USE_YCOCG
	return vec4(YCoCg_RGB(c.rgb).rgb, c.a);
#else
	return c;
#endif
}

// ------------------------------------------------------------------

vec3 tonemap_reinhard_simple(vec3 color)
{
	return color / (color + 1);
}

// ------------------------------------------------------------------

vec3 inverse_tonemap_reinhard_simple(vec3 color)
{
	return color / (1 - color);
}

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
	vec3 resolved;
    
#ifdef UNJITTER_TEX_COORDS
    vec2 uv = PS_IN_TexCoord - current_prev_jitter.xy;
#else
    vec2 uv = PS_IN_TexCoord;
#endif

    // Gather 3x3 neighbourhood texels.
    vec3 neighbourhood[9];
        
    neighbourhood[0] = sample_color(s_Color, uv + vec2(-1, -1) * u_PixelSize ).xyz;
    neighbourhood[1] = sample_color(s_Color, uv + vec2(+0, -1) * u_PixelSize ).xyz;
    neighbourhood[2] = sample_color(s_Color, uv + vec2(+1, -1) * u_PixelSize ).xyz;
    neighbourhood[3] = sample_color(s_Color, uv + vec2(-1, +0) * u_PixelSize ).xyz;
    neighbourhood[4] = sample_color(s_Color, uv + vec2(+0, +0) * u_PixelSize ).xyz;
    neighbourhood[5] = sample_color(s_Color, uv + vec2(+1, +0) * u_PixelSize ).xyz;
    neighbourhood[6] = sample_color(s_Color, uv + vec2(-1, +1) * u_PixelSize ).xyz;
    neighbourhood[7] = sample_color(s_Color, uv + vec2(+0, +1) * u_PixelSize ).xyz;
    neighbourhood[8] = sample_color(s_Color, uv + vec2(+1, +1) * u_PixelSize ).xyz;

    // Find minimum, maximum and average from the 3x3 neighbourhood.
    vec3 n_min = neighbourhood[0];
    vec3 n_max = neighbourhood[0]; 
    vec3 avg = neighbourhood[0];

    for(int i = 1; i < 9; ++i) 
	{
        n_min = min(n_min, neighbourhood[i]);
        n_max = max(n_max, neighbourhood[i]);
        avg += neighbourhood[i];
    }

    avg /= 9;

    // Sample velocity from the velocity buffer (velocity direction is from previous->current).
#ifdef VELOCITY_DILATION

#else
    vec2 velocity = texture(s_Velocity, uv).ba;
#endif
    
    vec2 history_uv = PS_IN_TexCoord - velocity.xy;
    
    // Sample from history buffer
    vec3 history = sample_color(s_History, history_uv).xyz; 

#ifdef CLIP_AABB

#else
    // Clamp history to neighbourhood.  
    history = clamp(history , n_min, n_max);
#endif

    // Sample current texel.
    vec3 current = neighbourhood[4];

    // Blend factor
    float blend = 0.05;
    
    bvec2 a = greaterThan(history_uv, vec2(1.0, 1.0));
    bvec2 b = lessThan(history_uv, vec2(0.0, 0.0));

    // if history sample is outside screen, switch to aliased image as a fallback.
    blend = any(bvec2(any(a), any(b))) ? 1.0 : blend;
    
#ifdef HDR_COLOR_BUFFER
    // Tonemap prior to blending to reduce flickering when using HDR color buffers.
    current = tonemap_reinhard_simple(current);
    history = tonemap_reinhard_simple(history);
#endif

    // Finally, blend current and clamped history sample.
    resolved = mix(history, current, vec3(blend));

#ifdef HDR_COLOR_BUFFER
    // Apply inverse tonemapping to get back into HDR.
    resolved = inverse_tonemap_reinhard_simple(resolved);
#endif

	PS_OUT_FragColor = resolve_color(vec4(resolved, 1.0)).xyz;
}

// ------------------------------------------------------------------