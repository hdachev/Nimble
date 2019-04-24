#ifndef DEPTH_CONVERSION_GLSL
#define DEPTH_CONVERSION_GLSL

// ------------------------------------------------------------------
// Helper functions for converting between different depth representations
// ------------------------------------------------------------------

// Convert an exponential depth value from the current views' projection to linear 0..1 depth
float linear_01_depth(float z)
{
    return 1.0 / (z_buffer_params.x * z + z_buffer_params.y);
}

// ------------------------------------------------------------------

// Convert an exponential depth value from the current views' projection to linear view-space depth
float linear_eye_depth(float z)
{
    return 1.0 / (z_buffer_params.z * z + z_buffer_params.w);
}

// ------------------------------------------------------------------

// Convert an exponential depth value from an arbitrary views' projection to linear 0..1 depth
float exp_01_to_linear_01_depth(float z, float n, float f)
{
    float z_buffer_params_y = f / n;
    float z_buffer_params_x = 1.0 - z_buffer_params_y;

    return 1.0 / (z_buffer_params_x * z + z_buffer_params_y);
}

// ------------------------------------------------------------------

// Convert an exponential depth value from an arbitrary views' projection to linear view-space depth
float exp_01_to_linear_eye_depth(float z, float n, float f)
{
    float z_buffer_params_y = f / n;
    float z_buffer_params_x = 1.0 - z_buffer_params_y;
    float z_buffer_params_z = z_buffer_params_x / f;
    float z_buffer_params_w = z_buffer_params_y / f;

    return 1.0 / (z_buffer_params_z * z + z_buffer_params_w);
}

// ------------------------------------------------------------------

#endif