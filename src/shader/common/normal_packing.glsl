#ifndef NORMAL_PACKING_GLSL
#define NORMAL_PACKING_GLSL

// ------------------------------------------------------------------
// Packing and unpacking Normal vectors into a G-Buffer
// https://aras-p.info/texts/CompactNormalStorage.html
// ------------------------------------------------------------------

#if defined(NORMAL_PACKING_RECONSTRUCT_Z)

// Method #1: store X&Y, reconstruct Z

vec2 encode_normal(vec3 n)
{
    return vec2(n.xy * 0.5 + 0.5);
}

// ------------------------------------------------------------------

vec3 decode_normal(vec2 enc)
{
    vec3 n;
    n.xy = enc * 2-1;
    n.z = sqrt(1 - dot(n.xy, n.xy));
    return n;
}

// ------------------------------------------------------------------

#elif defined(NORMAL_PACKING_SPHERICAL_COORDINATES)

// Method #3: Spherical Coordinates

#ifndef kPI
    #define kPI 3.1415926536f
#endif

vec2 encode_normal(vec3 n)
{
    return (vec2(atan2(n.y, n.x) / kPI, n.z) + 1.0) * 0.5;
}

// ------------------------------------------------------------------

vec3 decode_normal(vec2 enc)
{
    vec2 ang = enc * 2 - 1;
    vec2 scth = vec2(sin(ang.x), cos(ang.x));
    vec2 scphi = vec2(sqrt(1.0 - ang.y * ang.y), ang.y);
    return vec3(scth.y * scphi.x, scth.x * scphi.x, scphi.y);
}

// ------------------------------------------------------------------

#elif defined(NORMAL_PACKING_SPHEREMAP_TRANSFORM) // View-space only

// Method #4: Spheremap Transform

vec2 encode_normal(vec3 n)
{
    float f = sqrt(8.0 * n.z + 8.0);
    return n.xy / f + 0.5;
}

// ------------------------------------------------------------------

vec3 decode_normal(vec2 enc)
{
    vec2 fenc = enc * 4.0 - 2.0;
    float f = dot(fenc, fenc);
    float g = sqrt(1.0 - f / 4.0);
    vec3 n;
    n.xy = fenc * g;
    n.z = 1 - f / 2.0;
    return n;
}

// ------------------------------------------------------------------

#else

// Method #7: Stereographic Projection

vec2 encode_normal(vec3 n)
{
    float scale = 1.7777;
    vec2 enc = n.xy / (n.z + 1);
    enc /= scale;
    enc = enc * 0.5 + 0.5;
    return vec2(enc);
}

// ------------------------------------------------------------------

vec3 decode_normal(vec2 enc)
{
    float scale = 1.7777;
    vec3 nn = vec3(enc.xy, 0.0) * vec3(2 * scale, 2 * scale, 0.0) + vec3(-scale, -scale, 1);
    float g = 2.0 / dot(nn.xyz, nn.xyz);
    vec3 n;
    n.xy = g * nn.xy;
    n.z = g - 1;
    return n;
}

#endif

// ------------------------------------------------------------------

#endif