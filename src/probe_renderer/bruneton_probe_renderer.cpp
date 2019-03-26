#include "bruneton_probe_renderer.h"
#include "../renderer.h"
#include "../resource_manager.h"
#include "../logger.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <gtc/matrix_transform.hpp>

namespace nimble
{
// -----------------------------------------------------------------------------------------------------------------------------------

BrunetonProbeRenderer::BrunetonProbeRenderer()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

BrunetonProbeRenderer::~BrunetonProbeRenderer()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool BrunetonProbeRenderer::initialize(Renderer* renderer, ResourceManager* res_mgr)
{
    register_float_parameter("Sun Radius", m_sun_angular_radius);
    register_float_parameter("Exposure", m_exposure);

    // Values from "Reference Solar Spectral Irradiance: ASTM G-173", ETR column
    // (see http://rredc.nrel.gov/solar/spectra/am1.5/ASTMG173/ASTMG173.html),
    // summed and averaged in each bin (e.g. the value for 360nm is the average
    // of the ASTM G-173 values for all wavelengths between 360 and 370nm).
    // Values in W.m^-2.
    int lambda_min = 360;
    int lambda_max = 830;

    double kSolarIrradiance[] = {
        1.11776, 1.14259, 1.01249, 1.14716, 1.72765, 1.73054, 1.6887, 1.61253, 1.91198, 2.03474, 2.02042, 2.02212, 1.93377, 1.95809, 1.91686, 1.8298, 1.8685, 1.8931, 1.85149, 1.8504, 1.8341, 1.8345, 1.8147, 1.78158, 1.7533, 1.6965, 1.68194, 1.64654, 1.6048, 1.52143, 1.55622, 1.5113, 1.474, 1.4482, 1.41018, 1.36775, 1.34188, 1.31429, 1.28303, 1.26758, 1.2367, 1.2082, 1.18737, 1.14683, 1.12362, 1.1058, 1.07124, 1.04992
    };

    // Values from http://www.iup.uni-bremen.de/gruppen/molspec/databases/
    // referencespectra/o3spectra2011/index.html for 233K, summed and averaged in
    // each bin (e.g. the value for 360nm is the average of the original values
    // for all wavelengths between 360 and 370nm). Values in m^2.
    double kOzoneCrossSection[] = {
        1.18e-27, 2.182e-28, 2.818e-28, 6.636e-28, 1.527e-27, 2.763e-27, 5.52e-27, 8.451e-27, 1.582e-26, 2.316e-26, 3.669e-26, 4.924e-26, 7.752e-26, 9.016e-26, 1.48e-25, 1.602e-25, 2.139e-25, 2.755e-25, 3.091e-25, 3.5e-25, 4.266e-25, 4.672e-25, 4.398e-25, 4.701e-25, 5.019e-25, 4.305e-25, 3.74e-25, 3.215e-25, 2.662e-25, 2.238e-25, 1.852e-25, 1.473e-25, 1.209e-25, 9.423e-26, 7.455e-26, 6.566e-26, 5.105e-26, 4.15e-26, 4.228e-26, 3.237e-26, 2.451e-26, 2.801e-26, 2.534e-26, 1.624e-26, 1.465e-26, 2.078e-26, 1.383e-26, 7.105e-27
    };

    // From https://en.wikipedia.org/wiki/Dobson_unit, in molecules.m^-2.
    double kDobsonUnit = 2.687e20;
    // Maximum number density of ozone molecules, in m^-3 (computed so at to get
    // 300 Dobson units of ozone - for this we divide 300 DU by the integral of
    // the ozone density profile defined below, which is equal to 15km).
    double kMaxOzoneNumberDensity = 300.0 * kDobsonUnit / 15000.0;
    // Wavelength independent solar irradiance "spectrum" (not physically
    // realistic, but was used in the original implementation).
    double kConstantSolarIrradiance   = 1.5;
    double kTopRadius                 = 6420000.0;
    double kRayleigh                  = 1.24062e-6;
    double kRayleighScaleHeight       = 8000.0;
    double kMieScaleHeight            = 1200.0;
    double kMieAngstromAlpha          = 0.0;
    double kMieAngstromBeta           = 5.328e-3;
    double kMieSingleScatteringAlbedo = 0.9;
    double kMiePhaseFunctionG         = 0.8;
    double kGroundAlbedo              = 0.1;
    double max_sun_zenith_angle       = (m_use_half_precision ? 102.0 : 120.0) / 180.0 * M_PI;

    DensityProfileLayer* rayleigh_layer = new DensityProfileLayer("rayleigh", 0.0, 1.0, -1.0 / kRayleighScaleHeight, 0.0, 0.0);
    DensityProfileLayer* mie_layer      = new DensityProfileLayer("mie", 0.0, 1.0, -1.0 / kMieScaleHeight, 0.0, 0.0);

    // Density profile increasing linearly from 0 to 1 between 10 and 25km, and
    // decreasing linearly from 1 to 0 between 25 and 40km. This is an approximate
    // profile from http://www.kln.ac.lk/science/Chemistry/Teaching_Resources/
    // Documents/Introduction%20to%20atmospheric%20chemistry.pdf (page 10).
    std::vector<DensityProfileLayer*> ozone_density;
    ozone_density.push_back(new DensityProfileLayer("absorption0", 25000.0, 0.0, 0.0, 1.0 / 15000.0, -2.0 / 3.0));
    ozone_density.push_back(new DensityProfileLayer("absorption1", 0.0, 0.0, 0.0, -1.0 / 15000.0, 8.0 / 3.0));

    std::vector<double> wavelengths;
    std::vector<double> solar_irradiance;
    std::vector<double> rayleigh_scattering;
    std::vector<double> mie_scattering;
    std::vector<double> mie_extinction;
    std::vector<double> absorption_extinction;
    std::vector<double> ground_albedo;

    for (int l = lambda_min; l <= lambda_max; l += 10)
    {
        double lambda = l * 1e-3; // micro-meters
        double mie    = kMieAngstromBeta / kMieScaleHeight * pow(lambda, -kMieAngstromAlpha);

        wavelengths.push_back(l);

        if (m_use_constant_solar_spectrum)
            solar_irradiance.push_back(kConstantSolarIrradiance);
        else
            solar_irradiance.push_back(kSolarIrradiance[(l - lambda_min) / 10]);

        rayleigh_scattering.push_back(kRayleigh * pow(lambda, -4));
        mie_scattering.push_back(mie * kMieSingleScatteringAlbedo);
        mie_extinction.push_back(mie);
        absorption_extinction.push_back(m_use_ozone ? kMaxOzoneNumberDensity * kOzoneCrossSection[(l - lambda_min) / 10] : 0.0);
        ground_albedo.push_back(kGroundAlbedo);
    }

    m_sky_model.m_half_precision              = m_use_half_precision;
    m_sky_model.m_combine_scattering_textures = m_use_combined_textures;
    m_sky_model.m_use_luminance               = m_use_luminance;
    m_sky_model.m_wave_lengths                = wavelengths;
    m_sky_model.m_solar_irradiance            = solar_irradiance;
    m_sky_model.m_sun_angular_radius          = m_sun_angular_radius;
    m_sky_model.m_bottom_radius               = m_bottom_radius;
    m_sky_model.m_top_radius                  = kTopRadius;
    m_sky_model.m_rayleigh_density            = rayleigh_layer;
    m_sky_model.m_rayleigh_scattering         = rayleigh_scattering;
    m_sky_model.m_mie_density                 = mie_layer;
    m_sky_model.m_mie_scattering              = mie_scattering;
    m_sky_model.m_mie_extinction              = mie_extinction;
    m_sky_model.m_mie_phase_function_g        = kMiePhaseFunctionG;
    m_sky_model.m_absorption_density          = ozone_density;
    m_sky_model.m_absorption_extinction       = absorption_extinction;
    m_sky_model.m_ground_albedo               = ground_albedo;
    m_sky_model.m_max_sun_zenith_angle        = max_sun_zenith_angle;
    m_sky_model.m_length_unit_in_meters       = m_length_unit_in_meters;

    int num_scattering_orders = 4;

    bool status = m_sky_model.initialize(num_scattering_orders, renderer, res_mgr);

    glm::mat4 capture_projection = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 10.0f);
    glm::mat4 capture_views[]    = {
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f)),
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, -1.0f, 0.0f))
    };

    for (int i = 0; i < 6; i++)
        m_cubemap_views[i] = capture_projection * capture_views[i];

    m_sun_dir = glm::vec3(0.0f);

    std::vector<std::string> defines;

    if (m_use_luminance == LUMINANCE::NONE)
        defines.push_back("RADIANCE_API_ENABLED");

    if (m_use_combined_textures)
        defines.push_back("COMBINED_SCATTERING_TEXTURES");

    m_env_map_vs = res_mgr->load_shader("shader/bruneton_sky_model/env_map_vs.glsl", GL_VERTEX_SHADER, defines);
    m_env_map_fs = res_mgr->load_shader("shader/bruneton_sky_model/env_map_fs.glsl", GL_FRAGMENT_SHADER, defines);

    if (m_env_map_vs && m_env_map_fs)
    {
        m_env_map_program = renderer->create_program(m_env_map_vs, m_env_map_fs);

        if (!m_env_map_program)
        {
            NIMBLE_LOG_ERROR("Failed to create program");
            return false;
        }
    }
    else
    {
        NIMBLE_LOG_ERROR("Failed to load shaders");
        return false;
    }

    return status;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void BrunetonProbeRenderer::env_map(double delta, Renderer* renderer, Scene* scene)
{
    uint32_t          num_lights = scene->directional_light_count();
    DirectionalLight* lights     = scene->directional_lights();

    if (num_lights > 0)
    {
        DirectionalLight& light   = lights[0];
        glm::vec3         sun_dir = light.transform.forward();

        if (m_sun_dir != sun_dir)
        {
            m_sun_dir = sun_dir;

            for (int i = 0; i < 6; i++)
                m_cubemap_rtv[i] = RenderTargetView(i, 0, 0, scene->env_map());

            glDisable(GL_DEPTH_TEST);
            glDisable(GL_CULL_FACE);

            m_env_map_program->use();

            m_sky_model.bind_rendering_uniforms(m_env_map_program.get());

            m_env_map_program->set_uniform("sun_size", glm::vec2(tan(m_sun_angular_radius), cos(m_sun_angular_radius)));
            m_env_map_program->set_uniform("sun_direction", -light.transform.forward());
            m_env_map_program->set_uniform("earth_center", glm::vec3(0.0f, -m_bottom_radius / m_length_unit_in_meters, 0.0f));
            m_env_map_program->set_uniform("exposure", m_exposure);

            for (int i = 0; i < 6; i++)
            {
                m_env_map_program->set_uniform("view_projection", m_cubemap_views[i]);

                renderer->bind_render_targets(1, &m_cubemap_rtv[i], nullptr);
                glViewport(0, 0, ENVIRONMENT_MAP_SIZE, ENVIRONMENT_MAP_SIZE);

                glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
                glClear(GL_COLOR_BUFFER_BIT);

                renderer->cube_vao()->bind();

                glDrawArrays(GL_TRIANGLES, 0, 36);
            }
        }
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void BrunetonProbeRenderer::diffuse(double delta, Renderer* renderer, Scene* scene)
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void BrunetonProbeRenderer::specular(double delta, Renderer* renderer, Scene* scene)
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::string BrunetonProbeRenderer::probe_contribution_shader_path()
{
    return "";
}

// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble