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

uniform sampler2D s_Texture;
uniform vec2 u_PixelSize;
uniform float u_Strength;

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
	vec2 half_pixel = u_PixelSize;
    vec2 uv = FS_IN_TexCoord;
	
	vec4 sum = vec4(0.0, 0.0, 0.0, 0.0);
	
	sum += (2.0 / 16.0) * texture(s_Texture, uv + vec2(-half_pixel.x , 0.0));
    sum += (2.0 / 16.0) * texture(s_Texture, uv + vec2(0.0, half_pixel.y));
    sum += (2.0 / 16.0) * texture(s_Texture, uv + vec2(half_pixel.x , 0.0));
    sum += (2.0 / 16.0) * texture(s_Texture, uv + vec2(0.0, -half_pixel.y));
    
	sum += (1.0 / 16.0) * texture(s_Texture, uv + vec2(-half_pixel.x, -half_pixel.y));
	sum += (1.0 / 16.0) * texture(s_Texture, uv + vec2(-half_pixel.x, half_pixel.y));
    sum += (1.0 / 16.0) * texture(s_Texture, uv + vec2(half_pixel.x, -half_pixel.y));
    sum += (1.0 / 16.0) * texture(s_Texture, uv + vec2(half_pixel.x, half_pixel.y));

    sum += (4.0 / 16.0) * texture(s_Texture, uv);


	FS_OUT_FragColor = u_Strength * sum.rgb;
}

// ------------------------------------------------------------------