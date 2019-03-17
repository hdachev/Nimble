#include <../../common/uniforms.glsl>
#include <../../common/helper.glsl>

#define DEPTH_LOD 1.0

// ------------------------------------------------------------------
// OUTPUTS ----------------------------------------------------------
// ------------------------------------------------------------------

out vec3 FS_OUT_FragColor;

// ------------------------------------------------------------------
// INPUTS -----------------------------------------------------------
// ------------------------------------------------------------------

in vec2 FS_IN_TexCoord;

// ------------------------------------------------------------------
// UNIFORMS ---------------------------------------------------------
// ------------------------------------------------------------------

uniform sampler2D s_Depth;
uniform sampler2D s_Dither;

uniform int u_NumSamples;
uniform vec4 u_MieG;
uniform int u_Dither;

// ------------------------------------------------------------------
// FUNCTIONS --------------------------------------------------------
// ------------------------------------------------------------------

float mie_scattering(float cos_angle)
{    
	return u_MieG.w * (u_MieG.x / (pow(u_MieG.y - u_MieG.z * cos_angle, 1.5)));			
}

// ------------------------------------------------------------------

float attenuation(vec3 frag_pos, int shadow_map_idx, int light_idx)
{
	int start_idx = shadow_map_idx * num_cascades; // Starting from this value
	int end_idx = start_idx + num_cascades; // Less that this value

	int index = start_idx;

	// TODO: Use spheres for shadow cascades for stability and ease of checking
    vec4 clip_pos = view_proj * vec4(frag_pos, 1.0);
	clip_pos /= clip_pos.w;
	float frag_depth = clip_pos.z * 0.5 + 0.5;
    
	// Find shadow cascade.
	for (int i = start_idx; i < (end_idx - 1); i++)
	{
		if (frag_depth > cascade_far_plane[i])
			index = i + 1;
	}

	// Transform frag position into Light-space.
	vec4 light_space_pos = cascade_matrix[index] * vec4(frag_pos, 1.0);

	float current_depth = light_space_pos.z;

	float depth = texture(s_DirectionalLightShadowMaps, vec3(light_space_pos.xy, float(index))).r; 
	
	return current_depth > depth ? 0.0 : 1.0;        
}

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
	if (directional_light_count > 0)
	{
		if (directional_light_casts_shadow[0] == 1)
		{
			float depth = textureLod(s_Depth, FS_IN_TexCoord, DEPTH_LOD).r;
			vec3 frag_pos = world_position_from_depth(FS_IN_TexCoord, depth);

			vec3 direction = view_pos.xyz - frag_pos;
			float march_distance = length(direction);
			direction = normalize(direction);
			float step_size = march_distance / u_NumSamples;

			#ifdef DITHER_8_8
				vec2 interleaved_pos = (mod(floor(FS_IN_TexCoord.xy), 8.0));
				float offset = texture(s_Dither, FS_IN_TexCoord + vec2(0.5 / 8.0, 0.5 / 8.0)).r;
			#else
				vec2 interleaved_pos = (mod(floor(FS_IN_TexCoord.xy), 4.0));
				float offset = texture(s_Dither, FS_IN_TexCoord + vec2(0.5 / 4.0, 0.5 / 4.0)).r;	
			#endif

			if (u_Dither == 0)
				offset = 0.0;

			vec3 current_pos = frag_pos + direction * step_size * offset;

			float cos_angle = dot(directional_light_direction[0].xyz, direction);
			vec3 v_light = vec3(0.0);

			for (int i = 0; i < u_NumSamples; i++)
			{
				float atten = attenuation(current_pos, 0, 0);
				v_light += atten;

				current_pos = current_pos + direction * step_size;
			}

			// Apply scattering
			v_light *= mie_scattering(cos_angle);

			// Apply light color
			v_light *= directional_light_color_intensity[0].xyz;

			// Divide by the number of samples
			v_light /= u_NumSamples;

			FS_OUT_FragColor = v_light * 2.0;
		}
		else
			FS_OUT_FragColor = vec3(0.0);
	}
	else
		FS_OUT_FragColor = vec3(0.0);
}

// ------------------------------------------------------------------