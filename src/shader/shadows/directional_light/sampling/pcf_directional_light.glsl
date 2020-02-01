// ------------------------------------------------------------------
// PCF  -------------------------------------------------------------
// ------------------------------------------------------------------

float depth_compare(float a, float b, float bias)
{
    return a - bias > b ? 1.0 : 0.0;
}

// ------------------------------------------------------------------

vec3 csm_debug_color(float frag_depth, uint light_idx)
{
	vec4 far_planes = directional_light_cascade_far_planes(light_idx);

	int index = 0;

	// Find shadow cascade.
	for (int i = 0; i < (num_cascades - 1); i++)
	{
		if (frag_depth > far_planes[i])
			index = i + 1;
	}

	if (index == 0)
		return vec3(1.0, 0.0, 0.0);
	else if (index == 1)
		return vec3(0.0, 1.0, 0.0);
	else if (index == 2)
		return vec3(0.0, 0.0, 1.0);
	else if (index == 3)
		return vec3(1.0, 1.0, 0.0);
	else
		return vec3(1.0, 0.0, 1.0);
}

// ------------------------------------------------------------------

#define PCF_FILTERING_GRID_49_SAMPLES

#if defined(PCF_FILTERING_GRID_9_SAMPLES)

float directional_light_shadow_test(uint idx, vec2 sdw_uv, float depth, float bias)
{
	float shadow = 0.0;
	vec2 texel_size = 1.0 / textureSize(s_DirectionalLightShadowMaps, 0).xy;

	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(-1.000000, -1.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(-1.000000, 0.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(-1.000000, 1.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(0.000000, -1.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(0.000000, 0.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(0.000000, 1.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(1.000000, -1.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(1.000000, 0.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(1.000000, 1.000000) * texel_size, float(idx), depth - bias));

	return shadow / 9.0;
}

// ------------------------------------------------------------------

#elif defined(PCF_FILTERING_GRID_25_SAMPLES)

float directional_light_shadow_test(uint idx, vec2 sdw_uv, float depth, float bias)
{
	float shadow = 0.0;
	vec2 texel_size = 1.0 / textureSize(s_DirectionalLightShadowMaps, 0).xy;

	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(-2.000000, -2.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(-2.000000, -1.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(-2.000000, 0.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(-2.000000, 1.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(-2.000000, 2.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(-1.000000, -2.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(-1.000000, -1.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(-1.000000, 0.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(-1.000000, 1.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(-1.000000, 2.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(0.000000, -2.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(0.000000, -1.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(0.000000, 0.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(0.000000, 1.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(0.000000, 2.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(1.000000, -2.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(1.000000, -1.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(1.000000, 0.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(1.000000, 1.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(1.000000, 2.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(2.000000, -2.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(2.000000, -1.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(2.000000, 0.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(2.000000, 1.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(2.000000, 2.000000) * texel_size, float(idx), depth - bias));

	return shadow / 25.0;
}

// ------------------------------------------------------------------

#elif defined(PCF_FILTERING_GRID_49_SAMPLES)

float directional_light_shadow_test(uint idx, vec2 sdw_uv, float depth, float bias)
{
	float shadow = 0.0;
	vec2 texel_size = 1.0 / textureSize(s_DirectionalLightShadowMaps, 0).xy;

	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(-3.000000, -3.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(-3.000000, -2.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(-3.000000, -1.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(-3.000000, 0.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(-3.000000, 1.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(-3.000000, 2.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(-3.000000, 3.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(-2.000000, -3.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(-2.000000, -2.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(-2.000000, -1.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(-2.000000, 0.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(-2.000000, 1.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(-2.000000, 2.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(-2.000000, 3.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(-1.000000, -3.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(-1.000000, -2.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(-1.000000, -1.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(-1.000000, 0.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(-1.000000, 1.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(-1.000000, 2.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(-1.000000, 3.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(0.000000, -3.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(0.000000, -2.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(0.000000, -1.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(0.000000, 0.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(0.000000, 1.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(0.000000, 2.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(0.000000, 3.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(1.000000, -3.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(1.000000, -2.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(1.000000, -1.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(1.000000, 0.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(1.000000, 1.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(1.000000, 2.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(1.000000, 3.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(2.000000, -3.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(2.000000, -2.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(2.000000, -1.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(2.000000, 0.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(2.000000, 1.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(2.000000, 2.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(2.000000, 3.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(3.000000, -3.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(3.000000, -2.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(3.000000, -1.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(3.000000, 0.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(3.000000, 1.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(3.000000, 2.000000) * texel_size, float(idx), depth - bias));
	shadow += texture(s_DirectionalLightShadowMaps, vec4(sdw_uv + vec2(3.000000, 3.000000) * texel_size, float(idx), depth - bias));

	return shadow / 49.0;
}

#endif

// ------------------------------------------------------------------

float directional_light_shadows(in FragmentProperties f, uint light_idx)
{
	int index = 0;
    float blend = 0.0;
    
	vec4 far_planes = directional_light_cascade_far_planes(light_idx);
    
	// Find shadow cascade.
	for (int i = 0; i < (num_cascades - 1); i++)
	{
		if (f.FragDepth > far_planes[i])
			index = i + 1;
	}

	int shadow_matrix_idx = directional_light_first_shadow_matrix_index(light_idx) + index;
	int shadow_map_idx = directional_light_first_shadow_map_index(light_idx) + index;

	blend = clamp( (f.FragDepth - far_planes[index] * 0.995) * 200.0, 0.0, 1.0);
    
    // Apply blend options.
    //blend *= options.z;

	// Transform frag position into Light-space.
	vec4 light_space_pos = shadow_matrices[shadow_matrix_idx] * vec4(f.Position, 1.0);

	float current_depth = light_space_pos.z;
    
	vec3 n = f.Normal;
	vec3 l = directional_light_direction(light_idx);
	float bias = max(0.0005 * (1.0 - dot(n, l)), 0.0005);  

	return directional_light_shadow_test(shadow_map_idx, light_space_pos.xy, current_depth, bias);

    // if (options.x == 1.0)
    // {
    //     //if (blend > 0.0 && index != num_cascades - 1)
    //     //{
    //     //    light_space_pos = texture_matrices[index + 1] * vec4(PS_IN_WorldFragPos, 1.0f);
    //     //    shadow_map_depth = texture(s_ShadowMap, vec3(light_space_pos.xy, float(index + 1))).r;
    //     //    current_depth = light_space_pos.z;
    //     //    float next_shadow = depth_compare(current_depth, shadow_map_depth, bias);
    //     //    
    //     //    return (1.0 - blend) * shadow + blend * next_shadow;
    //     //}
    //     //else
	// 		return (1.0 - shadow);
    // }
    // else
    //     return 0.0;

	// return 1.0;
}

// ------------------------------------------------------------------