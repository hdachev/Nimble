#include <../../common/uniforms.glsl>
#include <../../common/helper.glsl>

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
uniform sampler2D s_Normals;
uniform sampler2D s_MetalRough;
uniform sampler2D s_Depth;

// ------------------------------------------------------------------
// CONSTANTS --------------------------------------------------------
// ------------------------------------------------------------------

const float kRayStep = 0.1;
const float kMinRayStep = 0.1;
const int kMaxSteps = 30;
const float kSearchDist = 5.0;
const int kNumBinarySearchSteps = 5;
const float kReflectionSpecularFalloffExponent = 3.0;

// ------------------------------------------------------------------
// FUNCTIONS --------------------------------------------------------
// ------------------------------------------------------------------

vec3 binary_search(inout vec3 dir, inout vec3 hit_coord, inout float out_depth)
{
	float depth;
	vec4 projected_coord;

	for (int i = 0; i < kNumBinarySearchSteps; i++)
	{
		projected_coord = projMat * vec4(hit_coord, 1.0);
		projected_coord.xy /= projected_coord.w;
		projected_coord.xy = projected_coord.xy * 0.5 + 0.5;

		depth = get_view_space_depth(projected_coord.xy, texture(s_Depth, projected_coord.xy).r);

		out_depth = hit_coord.z - depth;

		dir *= 0.5;

		if (out_depth > 0.0)
			hit_coord += dir;
		else
			hit_coord -= dir;
	}

	projected_coord = projMat * vec4(hit_coord, 1.0);
	projected_coord.xy /= projected_coord.w;
	projected_coord.xy = projected_coord.xy * 0.5 + 0.5;

	return vec3(projected_coord.xy, depth);
}

// ------------------------------------------------------------------

vec3 ray_march(in vec3 dir, inout vec3 hit_coord, out float out_depth)
{
	dir *= kRayStep;

	float depth;
	vec4 projected_coord;

	for (int i = 0; i < kMaxSteps; i++)
	{
		hit_coord += dir;

		projected_coord = projMat * vec4(hit_coord, 1.0);
		projected_coord.xy /= projected_coord.w;
		projected_coord.xy = projected_coord.xy * 0.5 + 0.5;

		depth = get_view_space_depth(projected_coord.xy, texture(s_Depth, projected_coord.xy).r);

		if (depth > 1000.0)
			continue;

		out_depth = hit_coord.z - depth;

		if ((dir.z - out_depth) < 1.2)
		{
			if (out_depth <= 0.0)
			{
				vec3 result;
				result = binary_search(dir, hit_coord, out_depth);
				
				return result;
			}
		}
	}

	return vec3(projected_coord.xy, depth);
}

// ------------------------------------------------------------------

vec3 fresnel_schlick(float cos_theta, vec3 F0)
{	
	return F0 + (1.0 - F0) * pow(1.0 - cos_theta, 5.0);
}

// ------------------------------------------------------------------

#define Scale vec3(.8, .8, .8)
#define K 19.19

vec3 hash(vec3 a)
{
    a = fract(a * Scale);
    a += dot(a, a.yxz + K);
    return fract((a.xxy + a.yxx)*a.zyx);
}

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
	// Fetch depth from hardware depth buffer
	float depth = texture(s_Depth, PS_IN_TexCoord).r;

	// Reconstruct view space position
	vec3 view_pos = get_view_space_position(PS_IN_TexCoord, depth);

	// Get normal from G-Buffer in View Space
	vec3 view_normal = get_view_space_normal(PS_IN_TexCoord, s_Normals);

	// Retrieve metalness and roughness values
	float metallic = texture(s_MetalRough, PS_IN_TexCoord).r;
	float roughness = texture(s_MetalRough, PS_IN_TexCoord).g;

	// Albedo from previous frame
	vec3 albedo = texture(s_Color, PS_IN_TexCoord).rgb; 

	// Calculate reflection vector
	vec3 reflection = normalize(reflect(normalize(view_pos), view_normal));
	vec3 wp = vec3(invView * vec4(view_pos, 1.0));
    vec3 jitt = mix(vec3(0.0), vec3(hash(wp)), roughness);

	// Calculate fresnel
	vec3 F0 = vec3(0.03);
	F0 = mix(F0, albedo, metallic);
	vec3 fresnel = fresnel_schlick(max(dot(normalize(view_pos), view_normal), 0.0), F0);

	vec3 hit_pos = view_pos;
	float out_depth;
	vec3 ray = jitt + reflection * max(kMinRayStep, -view_pos.z);

	vec3 hit_coord = ray_march(ray, hit_pos, out_depth);

	vec2 tex_coord = smoothstep(0.2, 0.6, abs(vec2(0.5, 0.5) - hit_coord.xy));
	float screen_edge_factor = clamp(1.0 - (tex_coord.x + tex_coord.y), 0.0, 1.0);

	float reflection_multiplier = pow(metallic, kReflectionSpecularFalloffExponent) * screen_edge_factor * -reflection.z;

	vec3 ssr = texture(s_Color, hit_coord.xy).rgb * clamp(reflection_multiplier, 0.0, 0.9) * fresnel;

	PS_OUT_FragColor = ssr;
}

// ------------------------------------------------------------------