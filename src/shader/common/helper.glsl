// ------------------------------------------------------------------
// HELPER FUNCTIONS -------------------------------------------------
// ------------------------------------------------------------------

// https://aras-p.info/texts/CompactNormalStorage.html
// Method #4: Spheremap Transform

vec2 encode_normal(vec3 n)
{
    float f = sqrt(8.0 * n.z + 8.0);
    return n.xy / f + 0.5;
}

// ------------------------------------------------------------------

// https://aras-p.info/texts/CompactNormalStorage.html
// Method #4: Spheremap Transform

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

// Take exponential depth and convert into linear depth.

float get_linear_depth(sampler2D depth_sampler, vec2 tex_coord, float far, float near)
{
    float z = (2 * near) / (far + near - texture( depth_sampler, tex_coord ).x * (far - near));
    return z;
}

// ------------------------------------------------------------------

vec3 get_normal_from_map(vec3 tangent, vec3 bitangent, vec3 normal, vec2 tex_coord, sampler2D normal_map)
{
	// Create TBN matrix.
    mat3 TBN = mat3(normalize(tangent), normalize(bitangent), normalize(normal));

    // Sample tangent space normal vector from normal map and remap it from [0, 1] to [-1, 1] range.
    vec3 n = normalize(texture(normal_map, tex_coord).xyz * 2.0 - 1.0);

    // Multiple vector by the TBN matrix to transform the normal from tangent space to world space.
    n = normalize(TBN * n);

    return n;
}

// ------------------------------------------------------------------

vec2 parallax_occlusion_tex_coords(vec3 tangent_view_pos, vec3 tangent_frag_pos, vec2 tex_coord, float height_scale, sampler2D displacement_map)
{
    vec3 view_dir   = normalize(tangent_view_pos - tangent_frag_pos);
    float height = texture(displacement_map, tex_coord).r;
    vec2 p = view_dir.xy / view_dir.z * (height * height_scale);
    return tex_coord - p;
}

// ------------------------------------------------------------------