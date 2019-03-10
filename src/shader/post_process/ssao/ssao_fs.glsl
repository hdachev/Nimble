#include <../../common/uniforms.glsl>
#include <../../common/helper.glsl>

#define DEPTH_LOD 1

// ------------------------------------------------------------------
// OUTPUTS ----------------------------------------------------------
// ------------------------------------------------------------------

out float FS_OUT_FragColor;

// ------------------------------------------------------------------
// INPUTS -----------------------------------------------------------
// ------------------------------------------------------------------

in vec2 FS_IN_TexCoord;

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

uniform sampler2D s_Normals; // Normals
uniform sampler2D s_Depth; // Depth
uniform sampler2D s_Noise; // SSAO Noise
uniform vec2 u_ViewportSize;
uniform int u_NumSamples;
uniform float u_Radius;
uniform float u_Bias;

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
    // Decode normal from G-Buffer in view-space
    vec3 normal = get_view_space_normal(FS_IN_TexCoord, s_Normals);

    // Sample depth at current fragment from hardware depth buffer
    float frag_depth = textureLod(s_Depth, FS_IN_TexCoord, DEPTH_LOD).r; 
    // Reconstruct view-space position
    vec3 position = get_view_space_position(FS_IN_TexCoord, frag_depth);    
    // SSAO Scale
    vec2 scale = vec2(u_ViewportSize.x / 4.0, u_ViewportSize.y / 4.0);   
    // Fetch random vector
    vec3 random = normalize(texture(s_Noise, FS_IN_TexCoord * scale).rgb);  
    // Construct view-space TBN matrix
    vec3 tangent = normalize(random - normal * dot(random, normal));
    vec3 bitangent = cross(normal, tangent);
    mat3 TBN = mat3(tangent, bitangent, normal);

    float occlusion = 0.0; 

    for (int i = 0; i < u_NumSamples; i++)
    {
        // Transform kernel sample from tangent space into view space
        vec3 ssao_sample = TBN * kernel[i].xyz; 
        // Add sample to fragment position and scale by radius
        ssao_sample = position + ssao_sample * u_Radius; 
        // Transform sample into clip space
        vec4 offset = vec4(ssao_sample, 1.0);
        offset = proj_mat * offset;  
        // Perspective division
        offset.xyz /= offset.w; 
        // Remap to the [0, 1] range
        offset.xyz = offset.xyz * 0.5 + 0.5;    
        // Use offset to sample depth texture
        float sample_depth = get_view_space_position(offset.xy, textureLod(s_Depth, offset.xy, DEPTH_LOD).r).z;  
        float range_check = smoothstep(0.0, 1.0, u_Radius / abs(position.z - sample_depth));
        occlusion += (sample_depth >= (ssao_sample.z + u_Bias) ? 1.0 : 0.0) * range_check;
    }   
    occlusion = 1.0 - (occlusion / float(u_NumSamples));
    FS_OUT_FragColor = occlusion;
}

// ------------------------------------------------------------------