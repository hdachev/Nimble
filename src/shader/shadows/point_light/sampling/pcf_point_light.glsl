// ------------------------------------------------------------------
// PCF  -------------------------------------------------------------
// ------------------------------------------------------------------

float point_light_shadows(in FragmentProperties f, int light_idx)
{
    vec3 frag_to_light = f.Position - point_light_position(light_idx);

    // now get current linear depth as the length between the fragment and light position
    float current_depth = (length(frag_to_light) - point_light_near_field(light_idx)) / point_light_far_field(light_idx) - point_light_near_field(light_idx));
    
    // now test for shadows
    float bias = point_light_shadow_bias(light_idx);
    int shadow_map_idx = point_light_shadow_map_index(light_idx);
 
    return texture(s_PointLightShadowMaps, vec4(frag_to_light, float(shadow_map_idx)), current_depth - bias);
}

// ------------------------------------------------------------------