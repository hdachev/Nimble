// ------------------------------------------------------------------
// PCF  -------------------------------------------------------------
// ------------------------------------------------------------------

float spot_light_shadows(vec3 position, int light_idx)
{
	// Transform frag position into Light-space.
	vec4 light_space_pos = spot_light_shadow_matrix[light_idx] * vec4(position, 1.0);

	 vec3 proj_coords = light_space_pos.xyz / light_space_pos.w;
    // transform to [0,1] range
    proj_coords = proj_coords * 0.5 + 0.5;
    // get closest depth value from light's perspective (using [0,1] range fragPosLight as coords)
    float closest_depth = texture(s_SpotLightShadowMaps, vec3(proj_coords.xy, float(light_idx))).r; 
    // get depth of current fragment from light's perspective
    float current_depth = proj_coords.z;
    // check whether current frag pos is in shadow
    float shadow = current_depth > closest_depth  ? 1.0 : 0.0;

    return 1.0 - shadow;
}

// ------------------------------------------------------------------