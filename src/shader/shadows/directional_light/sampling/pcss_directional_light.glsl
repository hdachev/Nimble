// ------------------------------------------------------------------
// PCF  -------------------------------------------------------------
// ------------------------------------------------------------------

float depth_compare(float a, float b, float bias)
{
    return a - bias > b ? 1.0 : 0.0;
}

// ------------------------------------------------------------------

vec3 csm_debug_color(float frag_depth, uint light_idx)
{
	vec4 far_planes = directional_light_cascade_far_planes(light_idx);

	int index = 0;

	// Find shadow cascade.
	for (int i = 0; i < (num_cascades - 1); i++)
	{
		if (frag_depth > far_planes[i])
			index = i + 1;
	}

	if (index == 0)
		return vec3(1.0, 0.0, 0.0);
	else if (index == 1)
		return vec3(0.0, 1.0, 0.0);
	else if (index == 2)
		return vec3(0.0, 0.0, 1.0);
	else if (index == 3)
		return vec3(1.0, 1.0, 0.0);
	else
		return vec3(1.0, 0.0, 1.0);
}

// ------------------------------------------------------------------

#define PCSS_DISK_SAMPLE_COUNT 64
// Fibonacci Spiral Disk Sampling Pattern
// https://people.irisa.fr/Ricardo.Marques/articles/2013/SF_CGF.pdf
//
// Normalized direction vector portion of fibonacci spiral can be baked into a LUT, regardless of sample_count.
// This allows us to treat the directions as a progressive sequence, using any sample_count in range [0, n <= LUT_LENGTH]
// the radius portion of spiral construction is coupled to sample count, but is fairly cheap to compute at runtime per sample.
// Generated (in javascript) with:
// var res = "";
// for (var i = 0; i < 64; ++i)
// {
//     var a = Math.PI * (3.0 - Math.sqrt(5.0));
//     var b = a / (2.0 * Math.PI);
//     var c = i * b;
//     var theta = (c - Math.floor(c)) * 2.0 * Math.PI;
//     res += "vec2 (" + Math.cos(theta) + ", " + Math.sin(theta) + "),\n";
// }

const vec2 kFibonacciSpiralDirection[PCSS_DISK_SAMPLE_COUNT] =
{
    vec2 (1, 0),
    vec2 (-0.7373688780783197, 0.6754902942615238),
    vec2 (0.08742572471695988, -0.9961710408648278),
    vec2 (0.6084388609788625, 0.793600751291696),
    vec2 (-0.9847134853154288, -0.174181950379311),
    vec2 (0.8437552948123969, -0.5367280526263233),
    vec2 (-0.25960430490148884, 0.9657150743757782),
    vec2 (-0.46090702471337114, -0.8874484292452536),
    vec2 (0.9393212963241182, 0.3430386308741014),
    vec2 (-0.924345556137805, 0.3815564084749356),
    vec2 (0.423845995047909, -0.9057342725556143),
    vec2 (0.29928386444487326, 0.9541641203078969),
    vec2 (-0.8652112097532296, -0.501407581232427),
    vec2 (0.9766757736281757, -0.21471942904125949),
    vec2 (-0.5751294291397363, 0.8180624302199686),
    vec2 (-0.12851068979899202, -0.9917081236973847),
    vec2 (0.764648995456044, 0.6444469828838233),
    vec2 (-0.9991460540072823, 0.04131782619737919),
    vec2 (0.7088294143034162, -0.7053799411794157),
    vec2 (-0.04619144594036213, 0.9989326054954552),
    vec2 (-0.6407091449636957, -0.7677836880006569),
    vec2 (0.9910694127331615, 0.1333469877603031),
    vec2 (-0.8208583369658855, 0.5711318504807807),
    vec2 (0.21948136924637865, -0.9756166914079191),
    vec2 (0.4971808749652937, 0.8676469198750981),
    vec2 (-0.952692777196691, -0.30393498034490235),
    vec2 (0.9077911335843911, -0.4194225289437443),
    vec2 (-0.38606108220444624, 0.9224732195609431),
    vec2 (-0.338452279474802, -0.9409835569861519),
    vec2 (0.8851894374032159, 0.4652307598491077),
    vec2 (-0.9669700052147743, 0.25489019011123065),
    vec2 (0.5408377383579945, -0.8411269468800827),
    vec2 (0.16937617250387435, 0.9855514761735877),
    vec2 (-0.7906231749427578, -0.6123030256690173),
    vec2 (0.9965856744766464, -0.08256508601054027),
    vec2 (-0.6790793464527829, 0.7340648753490806),
    vec2 (0.0048782771634473775, -0.9999881011351668),
    vec2 (0.6718851669348499, 0.7406553331023337),
    vec2 (-0.9957327006438772, -0.09228428288961682),
    vec2 (0.7965594417444921, -0.6045602168251754),
    vec2 (-0.17898358311978044, 0.9838520605119474),
    vec2 (-0.5326055939855515, -0.8463635632843003),
    vec2 (0.9644371617105072, 0.26431224169867934),
    vec2 (-0.8896863018294744, 0.4565723210368687),
    vec2 (0.34761681873279826, -0.9376366819478048),
    vec2 (0.3770426545691533, 0.9261958953890079),
    vec2 (-0.9036558571074695, -0.4282593745796637),
    vec2 (0.9556127564793071, -0.2946256262683552),
    vec2 (-0.50562235513749, 0.8627549095688868),
    vec2 (-0.2099523790012021, -0.9777116131824024),
    vec2 (0.8152470554454873, 0.5791133210240138),
    vec2 (-0.9923232342597708, 0.12367133357503751),
    vec2 (0.6481694844288681, -0.7614961060013474),
    vec2 (0.036443223183926, 0.9993357251114194),
    vec2 (-0.7019136816142636, -0.7122620188966349),
    vec2 (0.998695384655528, 0.05106396643179117),
    vec2 (-0.7709001090366207, 0.6369560596205411),
    vec2 (0.13818011236605823, -0.9904071165669719),
    vec2 (0.5671206801804437, 0.8236347091470047),
    vec2 (-0.9745343917253847, -0.22423808629319533),
    vec2 (0.8700619819701214, -0.49294233692210304),
    vec2 (-0.30857886328244405, 0.9511987621603146),
    vec2 (-0.4149890815356195, -0.9098263912451776),
    vec2 (0.9205789302157817, 0.3905565685566777)
};

// ------------------------------------------------------------------

vec2 compute_fibonacci_spiral_disk_sample(const in int sampleIndex, const in float diskRadius, const in float sample_count_inverse, const in float sample_count_bias)
{
    float sampleRadius = diskRadius * sqrt((float)sampleIndex * sample_count_inverse + sample_count_bias);
    vec2 sampleDirection = kFibonacciSpiralDirection[sampleIndex];
    return sampleDirection * sampleRadius;
}

// ------------------------------------------------------------------

float penumbra_size_punctual(float reciever, float blocker)
{
    return abs((reciever - blocker) / blocker);
}

// ------------------------------------------------------------------

float penumbra_size_directional(float Reciever, float blocker, float range_scale)
{
    return abs(Reciever - blocker) * range_scale;
}

// ------------------------------------------------------------------

bool blocker_search(in int shadow_map_idx, inout float average_blocker_depth, inout float num_blockers, float light_area, vec3 coord, vec2 sample_jitter, int sample_count)
{
    float blocker_sum = 0.0;
    float sample_count_inverse = rcp((float)sample_count);
    float sample_count_bias = 0.5 * sample_count_inverse;
    float dither_rotation = sample_jitter.x;

    for (int i = 0; i < sample_count && i < PCSS_DISK_SAMPLE_COUNT; ++i)
    {
        vec2 offset = compute_fibonacci_spiral_disk_sample(i, light_area, sample_count_inverse, sample_count_bias);
        offset = vec2(offset.x *  sample_jitter.y + offset.y * sample_jitter.x,
                       offset.x * -sample_jitter.x + offset.y * sample_jitter.y);

        float shadow_map_depth = texture(s_DirectionalLightShadowMaps, vec4(coord.xy + offset, float(shadow_map_idx)));

        if (shadow_map_depth < coord.z)
        {
            blocker_sum  += shadow_map_depth;
            num_blockers += 1.0;
        }
    }
    average_blocker_depth = blocker_sum / num_blockers;

    return num_blockers >= 1;
}

// ------------------------------------------------------------------

float directional_light_shadows(in FragmentProperties f, uint light_idx)
{
	int index = 0;
    float blend = 0.0;
    
	vec4 far_planes = directional_light_cascade_far_planes(light_idx);
    
	// Find shadow cascade.
	for (int i = 0; i < (num_cascades - 1); i++)
	{
		if (f.FragDepth > far_planes[i])
			index = i + 1;
	}

	int shadow_matrix_idx = directional_light_first_shadow_matrix_index(light_idx) + index;
	int shadow_map_idx = directional_light_first_shadow_map_index(light_idx) + index;

	blend = clamp( (f.FragDepth - far_planes[index] * 0.995) * 200.0, 0.0, 1.0);
    
    // Apply blend options.
    //blend *= options.z;

	// Transform frag position into Light-space.
	vec4 light_space_pos = shadow_matrices[shadow_matrix_idx] * vec4(f.Position, 1.0);

	float current_depth = light_space_pos.z;
    
	vec3 n = f.Normal;
	vec3 l = directional_light_direction(light_idx);
	float bias = max(0.0005 * (1.0 - dot(n, l)), 0.0005);  

	return directional_light_shadow_test(shadow_map_idx, light_space_pos.xy, current_depth, bias);

    // if (options.x == 1.0)
    // {
    //     //if (blend > 0.0 && index != num_cascades - 1)
    //     //{
    //     //    light_space_pos = texture_matrices[index + 1] * vec4(PS_IN_WorldFragPos, 1.0f);
    //     //    shadow_map_depth = texture(s_ShadowMap, vec3(light_space_pos.xy, float(index + 1))).r;
    //     //    current_depth = light_space_pos.z;
    //     //    float next_shadow = depth_compare(current_depth, shadow_map_depth, bias);
    //     //    
    //     //    return (1.0 - blend) * shadow + blend * next_shadow;
    //     //}
    //     //else
	// 		return (1.0 - shadow);
    // }
    // else
    //     return 0.0;

	// return 1.0;
}

// ------------------------------------------------------------------