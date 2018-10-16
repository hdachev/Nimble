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

vec3 get_world_position(float ndc_depth)
{
	// Remap depth to [-1.0, 1.0] range. 
	float depth = ndc_depth * 2.0 - 1.0;

	// Take texture coordinate and remap to [-1.0, 1.0] range. 
	vec2 screen_pos = PS_IN_TexCoord * 2.0 - 1.0;

	// // Create NDC position.
	vec4 ndc_pos = vec4(screen_pos, depth, 1.0);

	// Transform back into world position.
	vec4 world_pos = invViewProj * ndc_pos;

	// Undo projection.
	world_pos = world_pos / world_pos.w;

	return world_pos.xyz;
}


// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
	// Fetch depth from hardware depth buffer
	float depth = texture(s_Depth, PS_IN_TexCoord).r;

	// Reconstruct world space position
	vec3 world_pos = get_world_position(PS_IN_TexCoord, depth);

	// Screen space position
	vec3 screen_space_position = vec3(PS_IN_TexCoord, depth) * 2.0 - vec3(1.0);;

	// Camera direction
	vec3 camera_dir = normalize(world_pos - viewPos);

	// Get normal from G-Buffer in World Space
	vec3 world_normal = texture(s_Normals, PS_IN_TexCoord);

	// Reflection vector
	vec3 reflection = normalize(reflect(camera_dir, world_normal));

	// Get a point along reflection vector to calculate screen space reflection vector
	vec4 point_along_reflection_vector = vec4(world_pos + reflection * 10.0, 1.0);
	vec4 screen_space_reflection_point = viewProj * point_along_reflection_vector;

	screen_space_reflection_point /= screen_space_reflection_point.w; 
	screen_space_reflection_point.xy = screen_space_reflection_point.xy * 0.5 + vec2(0.5);

	// Screen space reflection vector
	vec3 screen_space_reflection_vec = normalize(screen_space_reflection_point - screen_space_position);

	
}

// ------------------------------------------------------------------