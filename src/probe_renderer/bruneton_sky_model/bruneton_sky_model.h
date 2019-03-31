#pragma once

#include <string>
#include <vector>
#include <memory>
#include "../../ogl.h"

namespace nimble
{
class ResourceManager;
class Renderer;

class BrunetonSkyModel
{
private:
	//Dont change these
	const int NUM_THREADS = 8;
	const int READ = 0;
	const int WRITE = 1;
	const float SCALE = 1000.0f;

	//Will save the tables as 8 bit png files so they can be
	//viewed in photoshop. Used for debugging.
	const bool WRITE_DEBUG_TEX = false;

	//You can change these
	//The radius of the planet (Rg), radius of the atmosphere (Rt)
	const float Rg = 6360.0f;
	const float Rt = 6420.0f;
	const float RL = 6421.0f;

	//Dimensions of the tables
	const int TRANSMITTANCE_W = 256;
	const int TRANSMITTANCE_H = 64;

	const int IRRADIANCE_W = 64;
	const int IRRADIANCE_H = 16;

	const int INSCATTER_R = 32;
	const int INSCATTER_MU = 128;
	const int INSCATTER_MU_S = 32;
	const int INSCATTER_NU = 8;

	//Physical settings, Mie and Rayliegh values
	const float AVERAGE_GROUND_REFLECTANCE = 0.1f;
	const glm::vec4 BETA_R = glm::vec4(5.8e-3f, 1.35e-2f, 3.31e-2f, 0.0f);
	const glm::vec4 BETA_MSca = glm::vec4(4e-3f, 4e-3f, 4e-3f, 0.0f);
	const glm::vec4 BETA_MEx = glm::vec4(4.44e-3f, 4.44e-3f, 4.44e-3f, 0.0f);

	//Asymmetry factor for the mie phase function
	//A higher number meands more light is scattered in the forward direction
	const float MIE_G = 0.8f;

	//Half heights for the atmosphere air density (HR) and particle density (HM)
	//This is the height in km that half the particles are found below
	const float HR = 8.0f;
	const float HM = 1.2f;

	glm::vec3 m_beta_r = glm::vec3(0.0058f, 0.0135f, 0.0331f);
    float m_mie_g = 0.75f;
    float m_sun_intensity = 100.0f;

	Texture2D* m_transmittance_t;
	Texture2D* m_delta_et;
	Texture3D* m_delta_srt;
	Texture3D* m_delta_smt;
	Texture3D* m_delta_jt;
	Texture2D* m_irradiance_t[2];
	Texture3D* m_inscatter_t[2];

	std::shared_ptr<Shader> m_copy_inscatter_1_cs;
	std::shared_ptr<Shader> m_copy_inscatter_n_cs;
	std::shared_ptr<Shader> m_copy_irradiance_cs;
	std::shared_ptr<Shader> m_inscatter_1_cs;
	std::shared_ptr<Shader> m_inscatter_n_cs;
	std::shared_ptr<Shader> m_inscatter_s_cs;
	std::shared_ptr<Shader> m_irradiance_1_cs;
	std::shared_ptr<Shader> m_irradiance_n_cs;
	std::shared_ptr<Shader> m_transmittance_cs;

	std::shared_ptr<Program> m_copy_inscatter_1_program;
	std::shared_ptr<Program> m_copy_inscatter_n_program;
	std::shared_ptr<Program> m_copy_irradiance_program;
	std::shared_ptr<Program> m_inscatter_1_program;
	std::shared_ptr<Program> m_inscatter_n_program;
	std::shared_ptr<Program> m_inscatter_s_program;
	std::shared_ptr<Program> m_irradiance_1_program;
	std::shared_ptr<Program> m_irradiance_n_program;
	std::shared_ptr<Program> m_transmittance_program;

public:
	BrunetonSkyModel();
	~BrunetonSkyModel();

	bool initialize(Renderer* renderer, ResourceManager* res_mgr);
	void set_render_uniforms(Program* program, glm::vec3 direction);
	
private:
	void set_uniforms(Program* program);
	bool load_cached_textures();
	void write_textures();
	void precompute();
	Texture2D* new_texture_2d(int width, int height);
	Texture3D* new_texture_3d(int width, int height, int depth);
	void swap(Texture2D** arr);
	void swap(Texture3D** arr);
};
} // namespace nimble