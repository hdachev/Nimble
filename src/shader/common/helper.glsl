#include <depth_conversion.glsl>
#include <normal_packing.glsl>

#define UNJITTER_TEX_COORDS(tc) (tc - current_prev_jitter.xy)

// ------------------------------------------------------------------
// HELPER FUNCTIONS -------------------------------------------------
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

vec3 get_normal_from_map_ex(vec3 tangent, vec3 bitangent, vec3 normal, vec2 tex_coord, vec3 normal_from_map)
{
	// Create TBN matrix.
    mat3 TBN = mat3(normalize(tangent), normalize(bitangent), normalize(normal));

    vec3 n =  normalize(normal_from_map);

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

vec3 view_position_from_depth(vec2 tex_coords, float ndc_depth)
{
	// Remap depth to [-1.0, 1.0] range. 
	float depth = ndc_depth * 2.0 - 1.0;

	// Take texture coordinate and remap to [-1.0, 1.0] range. 
	vec2 screen_pos = tex_coords * 2.0 - 1.0;

	// // Create NDC position.
	vec4 ndc_pos = vec4(screen_pos, depth, 1.0);

	// Transform back into view position.
	vec4 view_pos = inv_proj * ndc_pos;

	// Undo projection.
	view_pos = view_pos / view_pos.w;

	return view_pos.xyz;
}

// ------------------------------------------------------------------

vec3 world_to_view_space_normal(vec3 n)
{
    return mat3(view_mat) * n;
}

// ------------------------------------------------------------------

vec3 view_to_world_space_normal(vec3 n)
{
    return mat3(inv_view) * n;
}

// ------------------------------------------------------------------

float luminance(vec3 color)
{
    return max(dot(color, vec3(0.299, 0.587, 0.114)), 0.0001);
}

// ------------------------------------------------------------------