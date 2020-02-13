#include <../common/uniforms.glsl>
#include <../tiled/common.glsl>

// ------------------------------------------------------------------
// SAMPLERS ---------------------------------------------------------
// ------------------------------------------------------------------

uniform samplerCube s_IrradianceMap;
uniform samplerCube s_PrefilteredMap;
uniform sampler2D s_BRDF;
uniform sampler2DArray s_ShadowMap;

uniform sampler2D s_GBufferRT1;
uniform sampler2D s_GBufferRT2;
uniform sampler2D s_GBufferRT3;
uniform sampler2D s_GBufferRT4;
uniform sampler2D s_Depth;
uniform sampler2D s_SSAO;

// ------------------------------------------------------------------
// INPUT VARIABLES  -------------------------------------------------
// ------------------------------------------------------------------

in vec2 FS_IN_TexCoord;

#include <../common/helper.glsl>
#include <../common/material.glsl>
#include <../pbr/pbr.glsl>

// ------------------------------------------------------------------
// OUTPUT VARIABLES  ------------------------------------------------
// ------------------------------------------------------------------

layout (location = 0) out vec3 FS_OUT_Color;

// ------------------------------------------------------------------
// UNIFORMS ---------------------------------------------------------
// ------------------------------------------------------------------

layout(std430, binding = 3) buffer u_LightIndices
{
	uint indices[];
};

// ------------------------------------------------------------------

layout(std430, binding = 4) buffer u_LightGrid
{
	uvec4 light_grid[];
};

// ------------------------------------------------------------------

layout (std140, binding = 5) uniform u_ClusterData
{
	vec4 scale_bias_aabb_extent;
};

// ------------------------------------------------------------------

uint visible_point_light_count(uint cluster_idx)
{
	return light_grid[cluster_idx].x;
}

// ------------------------------------------------------------------

uint visible_spot_light_count(uint cluster_idx)
{
	return light_grid[cluster_idx].y;
}

// ------------------------------------------------------------------

uint visible_point_light_start_offset(uint cluster_idx)
{
	return light_grid[cluster_idx].z;
}

// ------------------------------------------------------------------

uint visible_spot_light_start_offset(uint cluster_idx)
{
	return light_grid[cluster_idx].w;
}

// ------------------------------------------------------------------
// FUNCTIONS --------------------------------------------------------
// ------------------------------------------------------------------

vec3 unpack_normal()
{
    return normalize(texture(s_GBufferRT2, FS_IN_TexCoord).rgb);
}

// ------------------------------------------------------------------

vec4 unpack_albedo()
{
    return vec4(texture(s_GBufferRT1, FS_IN_TexCoord).xyz, 1.0);
}

// ------------------------------------------------------------------

float unpack_metalness()
{
    return texture(s_GBufferRT4, FS_IN_TexCoord).x;
}

// ------------------------------------------------------------------

float unpack_roughness()
{
    return texture(s_GBufferRT4, FS_IN_TexCoord).y;
}

// ------------------------------------------------------------------

float unpack_depth()
{
	return texture(s_Depth, FS_IN_TexCoord).x;
}

// ------------------------------------------------------------------

float unpack_ssao()
{
	return texture(s_SSAO, FS_IN_TexCoord).x;
}

// ------------------------------------------------------------------

void fill_fragment_properties(inout FragmentProperties f)
{
	f.FragDepth = unpack_depth();
	f.Position = world_position_from_depth(FS_IN_TexCoord, f.FragDepth);
	f.TexCoords = FS_IN_TexCoord;
}

// ------------------------------------------------------------------

void fragment_func(inout MaterialProperties m)
{
	m.albedo = unpack_albedo();
	m.normal = unpack_normal();
	m.metallic = unpack_metalness();
	m.roughness = unpack_roughness();
}

// ------------------------------------------------------------------

uint cluster_z_id(in float view_z)
{
	return uint(max(log2(view_z) * scale_bias_aabb_extent.x + scale_bias_aabb_extent.y, 0.0));
}

// ------------------------------------------------------------------

uint cluster_index(vec4 frag_coord)
{
	uvec3 cluster_id  = uvec3(frag_coord.xy * scale_bias_aabb_extent.z, cluster_z_id(linear_eye_depth(frag_coord.z)));
    uint  cluster_idx = cluster_id.x +
						cluster_id.y * CLUSTER_GRID_DIM_X +
                   	    cluster_id.z * CLUSTER_GRID_DIM_X * CLUSTER_GRID_DIM_Y;

	return cluster_idx;
}

// ------------------------------------------------------------------

vec3 visible_light_contribution(in MaterialProperties m, in FragmentProperties f, in PBRProperties pbr)
{
	vec3 Lo = vec3(0.0);

#ifdef DIRECTIONAL_LIGHTS
	for (int i = 0; i < directional_light_count(); i++)
		Lo += pbr_directional_light_contribution(m, f, pbr, i);

#endif
    uint cluster_idx = cluster_index(vec4(gl_FragCoord.xy, f.FragDepth, gl_FragCoord.w));

#ifdef POINT_LIGHTS
	uint pl_offset = visible_point_light_start_offset(cluster_idx);
	uint pl_count = visible_point_light_count(cluster_idx);

	for (uint i = 0; i < pl_count; i++)
		Lo += pbr_point_light_contribution(m, f, pbr, int(indices[pl_offset + i]));
#endif

#ifdef SPOT_LIGHTS
	uint sl_offset = visible_spot_light_start_offset(cluster_idx);
	uint sl_count = visible_spot_light_count(cluster_idx);

	for (uint i = 0; i < sl_count; i++)
		Lo += pbr_spot_light_contribution(m, f, pbr, int(indices[sl_offset + i]));

#endif

	return Lo;
}

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
	FragmentProperties f;

	fill_fragment_properties(f);

	MaterialProperties m;

	// Set material properties
	fragment_func(m);

	PBRProperties pbr;

	// Set PBR properties
	pbr.N = m.normal;
	pbr.V = normalize(view_pos.xyz - f.Position); // FragPos -> ViewPos vector
	pbr.R = reflect(-pbr.V, pbr.N); 
	pbr.F0 = vec3(0.04);
	pbr.F0 = mix(pbr.F0, m.albedo.xyz, m.metallic);
	pbr.NdotV = max(dot(pbr.N, pbr.V), 0.0);
	pbr.F = fresnel_schlick_roughness(pbr.NdotV, pbr.F0, m.roughness);
	pbr.kS = pbr.F;
	pbr.kD = vec3(1.0) - pbr.kS;
	pbr.kD *= 1.0 - m.metallic;

	// Output radiance
	vec3 Lo = vec3(0.0);

	// Add all light contributions
	Lo += visible_light_contribution(m, f, pbr);

	vec3 irradiance = texture(s_IrradianceMap, pbr.N).rgb;
	vec3 diffuse = irradiance * m.albedo.xyz;

	// Sample prefilter map and BRDF LUT
	// vec3 prefilteredColor = textureLod(s_PrefilteredMap, pbr.R, m.roughness * kMaxLOD).rgb;
	// vec2 brdf = texture(s_BRDF, vec2(max(pbr.NdotV, 0.0), m.roughness)).rg;
	// vec3 specular = prefilteredColor * (pbr.F * brdf.x + brdf.y);

	// vec3 ambient = (pbr.kD * diffuse + specular) * kAmbient;
	float ambient = unpack_ssao();
	vec3 color = Lo + (m.albedo.xyz * ambient * 0.3);// + ambient;

	uint cluster_idx = cluster_index(vec4(gl_FragCoord.xy, f.FragDepth, gl_FragCoord.w));

	uint visible_light_count = visible_point_light_count(cluster_idx) + visible_spot_light_count(cluster_idx);
	float intensity = float(clamp(int(visible_light_count), 0, CLUSTER_DEBUG_MAX_LIGHTS)) / float(CLUSTER_DEBUG_MAX_LIGHTS);

	float minimum = 0.0;
	float maximum = 1.0;
	float ratio = 2 * (intensity - minimum) / (maximum - minimum);
	float b = max(0, 1 - ratio);
	float r = max(0, ratio - 1);
	float g = max(0, 1.0 - b - r);

    if (scale_bias_aabb_extent.w == 1.0)
		FS_OUT_Color = vec3(r,g,b);
	else
		FS_OUT_Color = color;
}

// ------------------------------------------------------------------