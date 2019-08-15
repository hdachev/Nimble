// ------------------------------------------------------------------
// PCF  -------------------------------------------------------------
// ------------------------------------------------------------------

float point_light_shadows(in FragmentProperties f, int shadow_map_idx, int light_idx)
{
    vec3 frag_to_light = f.Position - point_light_position_range[light_idx].xyz;

    // now get current linear depth as the length between the fragment and light position
    float current_depth = length(frag_to_light)/point_light_position_range[light_idx].w;
    
    // now test for shadows
    float bias = shadow_map_bias[light_idx].z;

    return texture(s_PointLightShadowMaps, vec4(frag_to_light, float(shadow_map_idx)), current_depth - bias);
}

// ------------------------------------------------------------------