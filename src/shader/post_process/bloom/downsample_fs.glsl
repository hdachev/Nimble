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

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
	vec2 half_pixel = 0.5 * vec2(u_PixelSize.x, u_PixelSize.y);
    vec2 one_pixel = 1.0 * vec2(u_PixelSize.x, u_PixelSize.y);
    
	vec2 uv = FS_IN_TexCoord;
	
	vec4 sum = vec4(0.0, 0.0, 0.0, 0.0);
	
	sum += (4.0/32.0) * texture(s_Texture, uv).rgba;
	
    sum += (4.0/32.0) * texture(s_Texture, uv + vec2(-half_pixel.x, -half_pixel.y));
	sum += (4.0/32.0) * texture(s_Texture, uv + vec2(+half_pixel.x, +half_pixel.y));
    sum += (4.0/32.0) * texture(s_Texture, uv + vec2(+half_pixel.x, -half_pixel.y));
    sum += (4.0/32.0) * texture(s_Texture, uv + vec2(-half_pixel.x, +half_pixel.y)	);
			
	sum += (2.0/32.0) * texture(s_Texture, uv + vec2(+one_pixel.x, 0.0));
	sum += (2.0/32.0) * texture(s_Texture, uv + vec2(-one_pixel.x, 0.0));
	sum += (2.0/32.0) * texture(s_Texture, uv + vec2(0.0, +one_pixel.y));
	sum += (2.0/32.0) * texture(s_Texture, uv + vec2(0.0, -one_pixel.y));
	
	sum += (1.0/32.0) * texture(s_Texture, uv + vec2(+one_pixel.x, +one_pixel.y));
	sum += (1.0/32.0) * texture(s_Texture, uv + vec2(-one_pixel.x, +one_pixel.y));
	sum += (1.0/32.0) * texture(s_Texture, uv + vec2(+one_pixel.x, -one_pixel.y));
	sum += (1.0/32.0) * texture(s_Texture, uv + vec2(-one_pixel.x, -one_pixel.y));

	FS_OUT_FragColor = sum.rgb;
}

// ------------------------------------------------------------------