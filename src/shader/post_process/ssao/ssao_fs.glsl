#include <../../common/uniforms.glsl>
#include <../../common/helper.glsl>

// ------------------------------------------------------------------
// OUTPUTS ----------------------------------------------------------
// ------------------------------------------------------------------

out float FragColor;

// ------------------------------------------------------------------
// INPUTS -----------------------------------------------------------
// ------------------------------------------------------------------

in vec2 PS_IN_TexCoord;

// ------------------------------------------------------------------
// UNIFORM BUFFERS --------------------------------------------------
// ------------------------------------------------------------------

layout (std140) uniform u_SSAOData
{
	vec4 kernel[64];
};

// ------------------------------------------------------------------
// UNIFORMS ---------------------------------------------------------
// ------------------------------------------------------------------

uniform sampler2D s_GBufferNormals; // Normals
uniform sampler2D s_GBufferRTDepth; // Depth
uniform sampler2D s_Noise; // SSAO Noise

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
    if (ssao == 1)
    {
        // Decode normal from G-Buffer in view-space
        vec3 normal = get_view_space_normal(PS_IN_TexCoord, s_GBufferNormals);
 
        // Sample depth at current fragment from hardware depth buffer
        float frag_depth = texture(s_GBufferRTDepth, PS_IN_TexCoord).r;

        // Reconstruct view-space position
        vec3 position = get_view_space_position(PS_IN_TexCoord, frag_depth);

        // SSAO Scale
        vec2 scale = vec2(float(viewport_width / 2.0) / 4.0, float(viewport_height / 2.0) / 4.0);

        // Fetch random vector
        vec3 random = normalize(texture(s_Noise, PS_IN_TexCoord * scale).rgb);

        // Construct view-space TBN matrix
        vec3 tangent = normalize(random - normal * dot(random, normal));
        vec3 bitangent = cross(normal, tangent);
        mat3 TBN = mat3(tangent, bitangent, normal);

        float occlusion = 0.0;

        for (int i = 0; i < ssao_num_samples; i++)
        {
            // Transform kernel sample from tangent space into view space
            vec3 ssao_sample = TBN * kernel[i].xyz;

            // Add sample to fragment position and scale by radius
            ssao_sample = position + ssao_sample * ssao_radius;

            // Transform sample into clip space
            vec4 offset = vec4(ssao_sample, 1.0);
            offset = projMat * offset;

            // Perspective division
            offset.xyz /= offset.w;

            // Remap to the [0, 1] range
            offset.xyz = offset.xyz * 0.5 + 0.5;

            // Use offset to sample depth texture
            float sample_depth = get_view_space_position(offset.xy, texture(s_GBufferRTDepth, offset.xy).r).z;

            float range_check = smoothstep(0.0, 1.0, ssao_radius / abs(position.z - sample_depth));
            occlusion += (sample_depth >= ssao_sample.z + ssao_bias ? 1.0 : 0.0) * range_check;
        }

        occlusion = 1.0 - (occlusion / float(ssao_num_samples));
        FragColor = occlusion;
    }
    else
        FragColor = 1.0;
}

// ------------------------------------------------------------------