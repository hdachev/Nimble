// ------------------------------------------------------------------
// PCF  -------------------------------------------------------------
// ------------------------------------------------------------------

float point_light_shadows(in FragmentProperties f, int shadow_map_idx, int light_idx)
{
    vec3 frag_to_light = f.Position - point_light_position_range[light_idx].xyz;
    // use the light to fragment vector to sample from the depth map    
    float closest_depth = texture(s_PointLightShadowMaps, vec4(frag_to_light, float(shadow_map_idx))).r;
    // it is currently in linear range between [0,1]. Re-transform back to original value
    closest_depth *= point_light_position_range[light_idx].w;
    // now get current linear depth as the length between the fragment and light position
    float current_depth = length(frag_to_light);
    // now test for shadows
    float bias = shadow_map_bias[light_idx].z;
    float shadow = current_depth -  bias > closest_depth ? 1.0 : 0.0;

    return 1.0 - shadow;
}

// ------------------------------------------------------------------