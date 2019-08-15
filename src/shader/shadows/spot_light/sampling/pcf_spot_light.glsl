// ------------------------------------------------------------------
// PCF  -------------------------------------------------------------
// ------------------------------------------------------------------

float spot_light_shadows(in FragmentProperties f, int shadow_map_idx, int light_idx)
{
	// Transform frag position into Light-space.
	vec4 light_space_pos = spot_light_shadow_matrix[shadow_map_idx] * vec4(f.Position, 1.0);

	vec2 proj_coords = light_space_pos.xy / light_space_pos.w;

    // get depth of current fragment from light's perspective
    float current_depth = light_space_pos.z;

    // transform to [0,1] range
    proj_coords = proj_coords * 0.5 + 0.5;

    // check whether current frag pos is in shadow
    float bias = shadow_map_bias[light_idx].y;
    current_depth = ((current_depth - bias)/light_space_pos.w) * 0.5 + 0.5;

    return texture(s_SpotLightShadowMaps, vec4(proj_coords.xy, float(shadow_map_idx), current_depth));
}

// ------------------------------------------------------------------