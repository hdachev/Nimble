// ------------------------------------------------------------------
// PCF  -------------------------------------------------------------
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