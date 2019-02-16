// ------------------------------------------------------------------
// PCF  -------------------------------------------------------------
// ------------------------------------------------------------------

float spot_light_shadows(vec3 position, int shadow_map_idx, int light_idx)
{
	// Transform frag position into Light-space.
	vec4 light_space_pos = spot_light_shadow_matrix[shadow_map_idx] * vec4(position, 1.0);

	 vec3 proj_coords = light_space_pos.xyz / light_space_pos.w;
    // transform to [0,1] range
    proj_coords = proj_coords * 0.5 + 0.5;
    // get closest depth value from light's perspective (using [0,1] range fragPosLight as coords)
    float closest_depth = texture(s_SpotLightShadowMaps, vec3(proj_coords.xy, float(shadow_map_idx))).r; 
    // get depth of current fragment from light's perspective
    float current_depth = proj_coords.z;
    // linearize depth values so that the bias can be applied
    float linear_closest_depth = depth_exp_to_linear_01(1.0, spot_light_direction_range[light_idx].w, closest_depth);
    float linear_current_depth = depth_exp_to_linear_01(1.0, spot_light_direction_range[light_idx].w, current_depth);
    // check whether current frag pos is in shadow
    float bias = 0.0005;
    float shadow = linear_current_depth - bias > linear_closest_depth  ? 1.0 : 0.0;

    return 1.0 - shadow;
}

// ------------------------------------------------------------------