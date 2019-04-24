// ------------------------------------------------------------------
// HELPER FUNCTIONS -------------------------------------------------
// ------------------------------------------------------------------

#define UNJITTER_TEX_COORDS(tc) (tc - current_prev_jitter.xy)

// Z buffer to linear 0..1 depth
float linear_01_depth(float z)
{
    return 1.0 / (z_buffer_params.x * z + z_buffer_params.y);
}

// ------------------------------------------------------------------

// Z buffer to linear depth
float linear_eye_depth(float z)
{
    return 1.0 / (z_buffer_params.z * z + z_buffer_params.w);
}

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

// Take exponential depth and convert into linear depth.

float depth_exp_to_view(mat4 inverse_proj, float exp_depth)
{
    exp_depth = exp_depth * 2.0 - 1.0;
    float w = inverse_proj[2][3] * exp_depth + inverse_proj[3][3];
    return (1.0 / w);
}

// ------------------------------------------------------------------

float depth_exp_to_view(float near, float far, float exp_depth)
{
   return (2.0 * near * far) / (far + near - exp_depth * (far - near));
}

// ------------------------------------------------------------------

float depth_view_to_linear_01(float near, float far, float depth)
{
	return (depth - near) / (far - near);
}

// ------------------------------------------------------------------

float depth_linear_01_to_view(float near, float far, float depth)
{
	return near + depth * (far - near);
}

// ------------------------------------------------------------------

float depth_exp_to_linear_01(float near, float far, float depth)
{
    float view_depth = depth_exp_to_view(near, far, depth);
    return depth_view_to_linear_01(near, far, view_depth);
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

vec2 parallax_occlusion(vec3 view_dir, vec2 tex_coord, float height_scale, sampler2D displacement_map)
{ 
    // number of depth layers
    const float minLayers = 8;
    const float maxLayers = 32;
    float numLayers = mix(maxLayers, minLayers, abs(dot(vec3(0.0, 0.0, 1.0), view_dir)));  
    // calculate the size of each layer
    float layerDepth = 1.0 / numLayers;
    // depth of current layer
    float currentLayerDepth = 0.0;
    // the amount to shift the texture coordinates per layer (from vector P)
    vec2 P = view_dir.xy / view_dir.z * height_scale; 
    vec2 deltaTexCoords = P / numLayers;
  
    // get initial values
    vec2  currentTexCoords     = tex_coord;
    float currentDepthMapValue = 1.0 - texture(displacement_map, currentTexCoords).r;
      
    while(currentLayerDepth < currentDepthMapValue)
    {
        // shift texture coordinates along direction of P
        currentTexCoords -= deltaTexCoords;
        // get depthmap value at current texture coordinates
        currentDepthMapValue = 1.0 - texture(displacement_map, currentTexCoords).r;  
        // get depth of next layer
        currentLayerDepth += layerDepth;  
    }
    
    // get texture coordinates before collision (reverse operations)
    vec2 prevTexCoords = currentTexCoords + deltaTexCoords;

    // get depth after and before collision for linear interpolation
    float afterDepth  = currentDepthMapValue - currentLayerDepth;
    float beforeDepth = (1.0 - texture(displacement_map, prevTexCoords).r) - currentLayerDepth + layerDepth;
 
    // interpolation of texture coordinates
    float weight = afterDepth / (afterDepth - beforeDepth);
    vec2 finalTexCoords = prevTexCoords * weight + currentTexCoords * (1.0 - weight);

    return finalTexCoords;
}

// ------------------------------------------------------------------

#ifdef DISPLACEMENT_TYPE_PARALLAX_OCCLUSION
vec2 get_parallax_occlusion_texcoords(vec2 tex_coord, vec3 tangent_view_pos, vec3 tangent_frag_pos)
{
	vec3 tangent_view_dir = normalize(tangent_view_pos - tangent_frag_pos);
	return parallax_occlusion(tangent_view_dir, tex_coord, 0.05, s_Displacement);
}
#endif

// ------------------------------------------------------------------

vec2 motion_vector(vec4 prev_pos, vec4 current_pos)
{
    // Perspective division.
    vec2 current = (current_pos.xy / current_pos.w);
    vec2 prev = (prev_pos.xy / prev_pos.w);

    // Remove jitter
    current -= current_prev_jitter.xy;
    prev -= current_prev_jitter.zw;

    // Remap to [0, 1] range
    current = current * 0.5 + 0.5;
    prev = prev * 0.5 + 0.5;

    // Calculate velocity (prev -> current)
    return (current - prev);
}

// ------------------------------------------------------------------

// GLSL version of log10 from HLSL
float log10(in float n) 
{
	const float kLogBase10 = 1.0 / log2( 10.0 );
	return log2( n ) * kLogBase10;
}

// ------------------------------------------------------------------

// GLSL version of log10 from HLSL
vec3 log10(in vec3 n) 
{
	return vec3(log10(n.x), log10(n.y), log10(n.z));
}

// ------------------------------------------------------------------

vec3 world_position_from_depth(vec2 tex_coords, float ndc_depth)
{
	// Remap depth to [-1.0, 1.0] range. 
	float depth = ndc_depth * 2.0 - 1.0;

	// Take texture coordinate and remap to [-1.0, 1.0] range. 
	vec2 screen_pos = tex_coords * 2.0 - 1.0;

	// // Create NDC position.
	vec4 ndc_pos = vec4(screen_pos, depth, 1.0);

	// Transform back into world position.
	vec4 world_pos = inv_view_proj * ndc_pos;

	// Undo projection.
	world_pos = world_pos / world_pos.w;

	return world_pos.xyz;
}

// ------------------------------------------------------------------

vec3 get_view_space_position(vec2 tex_coords, float depth)
{
    vec3 clip_space_position = vec3(tex_coords, depth) * 2.0 - vec3(1.0);

    //vec4 view_position = vec4(vec2(inv_proj[0][0], inv_proj[1][1]) * clip_space_position.xy, -1.0,
    //                               inv_proj[2][3] * clip_space_position.z + inv_proj[3][3]);
    vec4 view_position = inv_proj * vec4(clip_space_position, 1.0);
    return (view_position.xyz / view_position.w);
}

// ------------------------------------------------------------------

float get_view_space_depth(vec2 tex_coords, float depth)
{
    depth = depth * 2.0 - 1.0;
    float w = inv_proj[2][3] * depth + inv_proj[3][3];
    return (-1.0 / w);
}

// ------------------------------------------------------------------

vec3 get_view_space_normal(vec2 tex_coords, sampler2D g_buffer_normals)
{
	//vec2 encoded_normal = texture(g_buffer_normals, tex_coords).rg;
    //vec3 n = mat3(viewMat) * decode_normal(encoded_normal);
    vec3 n = mat3(view_mat) * normalize(texture(g_buffer_normals, tex_coords).rgb);
    return n;
}

// ------------------------------------------------------------------

float luminance(vec3 color)
{
    //return dot(vec3(0.2125, 0.7154, 0.0721), color);
    return max(dot(color, vec3(0.299, 0.587, 0.114)), 0.0001);
}

// ------------------------------------------------------------------