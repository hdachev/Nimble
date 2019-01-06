#include "material.h"
#include "ogl.h"

namespace nimble
{
	static uint32_t g_last_material_id = 0;

	// -----------------------------------------------------------------------------------------------------------------------------------

	Material::Material() : m_id(g_last_material_id++)
	{
		for (uint32_t i = 0; i < MAX_MATERIAL_TEXTURES; i++)
			m_custom_textures[i] = nullptr;

		for (uint32_t i = 0; i < MAX_MATERIAL_TEXTURES; i++)
			m_surface_textures[i] = nullptr;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	Material::~Material()
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Material::bind(Program* program, int32_t& unit)
	{
		// Bind surface textures
		bind_surface_textures(program, unit);

		// Bind custom textures
		bind_custom_textures(program, unit);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Material::bind_surface_texture(TextureType type, Program* program, int32_t& unit)
	{
		if (m_surface_textures[type])
		{
			if (program->set_uniform(kSurfaceTextureNames[type], unit))
				m_surface_textures[type]->bind(unit++);
		}
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Material::bind_surface_textures(Program* program, int32_t& unit)
	{
		for (uint32_t i = 0; i < (MAX_MATERIAL_TEXTURES - 1); i++)
			bind_surface_texture(static_cast<TextureType>(i), program, unit);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Material::bind_custom_textures(Program* program, int32_t& unit)
	{
		for (uint32_t i = 0; i < m_custom_texture_count; i++)
		{
			if (m_custom_textures[i])
			{
				if (program->set_uniform(kCustomTextureNames[i], unit))
					m_custom_textures[i]->bind(unit++);
			}
		}
	}

	// -----------------------------------------------------------------------------------------------------------------------------------
}