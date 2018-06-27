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
// HELPER FUNCTIONS -------------------------------------------------
// ------------------------------------------------------------------

vec3 get_view_space_position(vec2 tex_coords, sampler2D depth_buffer)
{
    vec2 ndc_pos = tex_coords * 2.0 - 1.0;
    float depth = texture(depth_buffer, tex_coords).r * 2.0 - 1.0;
    vec4 position = invProj * vec4(ndc_pos.x, ndc_pos.y, depth, 1.0);
    position = position / position.w;

    return position.xyz;
}

// ------------------------------------------------------------------

vec3 get_view_space_normal(vec2 tex_coords, sampler2D g_buffer_normals)
{
	vec2 encoded_normal = texture(g_buffer_normals, tex_coords).rg;
    vec3 n = mat3(viewMat) * decode_normal(encoded_normal);
    return n;
}

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
    if (ssao == 1)
    {
        // Decode normal from G-Buffer in view-space
        vec3 normal = get_view_space_normal(PS_IN_TexCoord, s_GBufferNormals);
 
        // Reconstruct view-space position
        vec3 position = get_view_space_position(PS_IN_TexCoord, s_GBufferRTDepth);

        // SSAO Scale
        vec2 scale = vec2(float(viewport_width) / 4.0, float(viewport_height) / 4.0);

        // Fetch random vector
        vec3 random = texture(s_Noise, PS_IN_TexCoord * scale).rgb;

        // Construct TBN matrix
        vec3 tangent = normalize(random - normal * dot(normal, random));
        vec3 bitangent = cross(normal, tangent);
        mat3 TBN = mat3(tangent, bitangent, normal);

        float occlusion = 0.0;

        for (int i = 0; i < u_NumSamples; i++)
        {
            vec3 sample = TBN * kernel[i];
            sample = position + sample * radius;
        }
    }
    else
        FragColor = 1.0;
}

// ------------------------------------------------------------------