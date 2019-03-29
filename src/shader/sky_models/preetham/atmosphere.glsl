// ------------------------------------------------------------------
// UNIFORMS ---------------------------------------------------------
// ------------------------------------------------------------------

uniform vec3 u_Direction;
uniform float u_PerezInvDen;
uniform int TABLE_SIZE;

uniform sampler2D s_Table;

// ------------------------------------------------------------------
// CONSTANTS --------------------------------------------------------
// ------------------------------------------------------------------

#define THETA_TABLE 0
#define GAMMA_TABLE 1

// XYZ/RGB for sRGB primaries
const vec3 kXYZToR = vec3( 3.2404542, -1.5371385, -0.4985314);
const vec3 kXYZToG = vec3(-0.9692660,  1.8760108,  0.0415560);
const vec3 kXYZToB = vec3( 0.0556434, -0.2040259,  1.0572252);
const vec3 kRGBToX = vec3(0.4124564,  0.3575761,  0.1804375);
const vec3 kRGBToY = vec3(0.2126729,  0.7151522,  0.0721750);
const vec3 kRGBToZ = vec3(0.0193339,  0.1191920,  0.9503041);

// ------------------------------------------------------------------
// FUNCTIONS --------------------------------------------------------
// ------------------------------------------------------------------

float table_value(int i, int comp)
{
    return texelFetch(s_Table, ivec2(i, comp), 0).r;
}

// ------------------------------------------------------------------

float map_gamma(float g)
{
    return sqrt(0.5f * (1.0f - g));
}

// ------------------------------------------------------------------

float table_lerp(float s, int n, int comp)
{
    s = clamp(s, 0.0, 1.0);
    
    s *= n - 1;

    int si0 = int(s);
    int si1 = (si0 + 1);
    float sf = s - si0;

    return table_value(si0, comp) * (1 - sf) + table_value(si1, comp) * sf;
}

// ------------------------------------------------------------------

vec3 xyY_to_XYZ(vec3 c)
{
    return vec3(c.x, c.y, 1.0 - c.x - c.y) * c.z / c.y;
}

// ------------------------------------------------------------------

vec3 xyY_to_RGB(vec3 xyY)
{
    vec3 XYZ = xyY_to_XYZ(xyY);
    return vec3(dot(kXYZToR, XYZ),
                dot(kXYZToG, XYZ),
                dot(kXYZToB, XYZ));
}

// ------------------------------------------------------------------

vec3 sky_rgb(vec3 v)
{
    float cosTheta = v.z;
    float cosGamma = dot(u_Direction, v);

    if (cosTheta < 0.0f)
        cosTheta = 0.0f;

    float t = cosTheta;
    float g = map_gamma(cosGamma);

    vec3 F = vec3(table_lerp(t, TABLE_SIZE, THETA_TABLE));
    vec3 G = vec3(table_lerp(g, TABLE_SIZE, GAMMA_TABLE));

#ifdef SIM_CLAMP
    F.z *= texelFetch(s_Table, ivec2(TABLE_SIZE, THETA_TABLE), 0).r;
    G.z *= texelFetch(s_Table, ivec2(TABLE_SIZE, GAMMA_TABLE), 0).r;
#endif

    vec3 xyY = (vec3(1.0) - F) * (vec3(1.0) + G);

    xyY *= u_PerezInvDen;

    return xyY_to_RGB(xyY);
}

// ------------------------------------------------------------------