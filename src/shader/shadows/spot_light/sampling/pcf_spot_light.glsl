// ------------------------------------------------------------------
// PCF  -------------------------------------------------------------
// ------------------------------------------------------------------

float spot_light_shadows(in FragmentProperties f, uint light_idx)
{
    int shadow_matrix_idx = spot_light_shadow_matrix_index(light_idx);

	// Transform frag position into Light-space.
	vec4 light_space_pos = shadow_matrices[shadow_matrix_idx] * vec4(f.Position, 1.0);

	vec2 proj_coords = light_space_pos.xy / light_space_pos.w;

    // get depth of current fragment from light's perspective
    float current_depth = light_space_pos.z;

    // transform to [0,1] range
    proj_coords = proj_coords * 0.5 + 0.5;

    // check whether current frag pos is in shadow
    float bias = spot_light_shadow_bias(light_idx);
    current_depth = ((current_depth - bias)/light_space_pos.w) * 0.5 + 0.5;

    int shadow_map_idx = spot_light_shadow_map_index(light_idx);

    return texture(s_SpotLightShadowMaps, vec4(proj_coords.xy, float(shadow_map_idx), current_depth));
}

// ------------------------------------------------------------------