#include <uniforms.glsl>
#include <utility.glsl>
#include <transmittance_functions.glsl>
#include <scattering_functions.glsl>
#include <irradiance_functions.glsl>
#include <rendering_functions.glsl>

// ------------------------------------------------------------------
// OUPUT ------------------------------------------------------------
// ------------------------------------------------------------------

out vec4 FS_OUT_Color;

// ------------------------------------------------------------------
// INPUT ------------------------------------------------------------
// ------------------------------------------------------------------

in vec3 FS_IN_WorldPos;

// ------------------------------------------------------------------
// UNIFORMS ---------------------------------------------------------
// ------------------------------------------------------------------

uniform vec3 sun_direction;
uniform vec2 sun_size;

uniform sampler2D transmittance_texture;
uniform sampler2D irradiance_texture;
uniform sampler3D scattering_texture;
uniform sampler3D single_mie_scattering_texture;

// ------------------------------------------------------------------
// FUNCTIONS --------------------------------------------------------
// ------------------------------------------------------------------

#ifdef RADIANCE_API_ENABLED

RadianceSpectrum GetSolarRadiance() 
{
	return solar_irradiance / (PI * sun_angular_radius * sun_angular_radius);
}

// ------------------------------------------------------------------

RadianceSpectrum GetSkyRadiance(
	Position camera, Direction view_ray, Length shadow_length,
	Direction sun_direction, out DimensionlessSpectrum transmittance) 
{
	return GetSkyRadiance(transmittance_texture,
		scattering_texture, single_mie_scattering_texture,
		camera, view_ray, shadow_length, sun_direction, transmittance);
}

// ------------------------------------------------------------------

#else

Luminance3 GetSolarRadiance()
{
	return solar_irradiance /
		(PI * sun_angular_radius * sun_angular_radius) *
		SUN_SPECTRAL_RADIANCE_TO_LUMINANCE;
}

// ------------------------------------------------------------------

Luminance3 GetSkyRadiance(
	Position camera, Direction view_ray, Length shadow_length,
	Direction sun_direction, out DimensionlessSpectrum transmittance) 
{
	return GetSkyRadiance(transmittance_texture,
		scattering_texture, single_mie_scattering_texture,
		camera, view_ray, shadow_length, sun_direction, transmittance) *
		SKY_SPECTRAL_RADIANCE_TO_LUMINANCE;
}

#endif

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
	vec3 view_direction = normalize(FS_IN_WorldPos);

	// Compute the radiance of the sky.
	vec3 transmittance;
	vec3 radiance = GetSkyRadiance(vec3(0.0, 0.0, 0.0), view_direction, 4.0, sun_direction, transmittance);

	// If the view ray intersects the Sun, add the Sun radiance.
	if (dot(view_direction, sun_direction) > sun_size.y) 
		radiance = radiance + transmittance * GetSolarRadiance();
		
	FS_OUT_Color = vec4(radiance, 1.0);
}

// ------------------------------------------------------------------