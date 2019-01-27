// ------------------------------------------------------------------
// PCF  -------------------------------------------------------------
// ------------------------------------------------------------------

float depth_compare(float a, float b, float bias)
{
    return a - bias > b ? 1.0 : 0.0;
}

// ------------------------------------------------------------------

vec3 debug_color(float frag_depth)
{
	int index = 0;

	// Find shadow cascade.
	for (int i = 0; i < num_cascades - 1; i++)
	{
		if (frag_depth > shadow_frustums[i].far_plane)
			index = i + 1;
	}

	if (index == 0)
		return vec3(1.0, 0.0, 0.0);
	else if (index == 1)
		return vec3(0.0, 1.0, 0.0);
	else if (index == 2)
		return vec3(0.0, 0.0, 1.0);
	else
		return vec3(1.0, 1.0, 0.0);
}

// ------------------------------------------------------------------

float directional_light_shadows(float frag_depth, vec3 position, vec3 n, vec3 l)
{
	// int index = 0;
    // float blend = 0.0;
    
	// // Find shadow cascade.
	// for (int i = 0; i < num_cascades - 1; i++)
	// {
	// 	if (frag_depth > shadow_frustums[i].far_plane)
	// 		index = i + 1;
	// }

	// blend = clamp( (frag_depth - shadow_frustums[index].far_plane * 0.995) * 200.0, 0.0, 1.0);
    
    // // Apply blend options.
    // //blend *= options.z;

	// // Transform frag position into Light-space.
	// vec4 light_space_pos = shadow_frustums[index].shadow_matrix * vec4(position, 1.0f);

	// float current_depth = light_space_pos.z;
    
	// float bias = max(0.0005 * (1.0 - dot(n, l)), 0.0005);  

	// float shadow = 0.0;
	// vec2 texelSize = 1.0 / textureSize(s_ShadowMap, 0).xy;
	// for(int x = -1; x <= 1; ++x)
	// {
	//     for(int y = -1; y <= 1; ++y)
	//     {
	//         float pcfDepth = texture(s_ShadowMap, vec3(light_space_pos.xy + vec2(x, y) * texelSize, float(index))).r; 
	//         shadow += current_depth - bias > pcfDepth ? 1.0 : 0.0;        
	//     }    
	// }
	// shadow /= 9.0;
	
	// return (1.0 - shadow);
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

	return 1.0;
}

// ------------------------------------------------------------------

float point_light_shadows(vec3 frag_to_light, int idx)
{
    // use the light to fragment vector to sample from the depth map    
    float closest_depth = texture(s_PointLightShadowMaps, vec4(frag_to_light, float(idx))).r;
    // it is currently in linear range between [0,1]. Re-transform back to original value
    closest_depth *= point_lights[idx].position_range.w;
    // now get current linear depth as the length between the fragment and light position
    float current_depth = length(frag_to_light);
    // now test for shadows
    float bias = 0.05; 
    float shadow = current_depth -  bias > closest_depth ? 1.0 : 0.0;

    return shadow;
}

// ------------------------------------------------------------------

float spot_light_shadows()
{
	return 1.0;
}

// ------------------------------------------------------------------