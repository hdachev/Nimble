#include "material.h"
#include "ogl.h"

namespace nimble
{
	// -----------------------------------------------------------------------------------------------------------------------------------

	Material::Material()
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

	void Material::bind(Program* program)
	{
		int32_t unit = 0;

		// Bind surface textures
		for (uint32_t i = 0; i < MAX_MATERIAL_TEXTURES; i++)
		{
			if (m_surface_textures[i])
			{
				if (program->set_uniform(kSurfaceTextureNames[i], unit))
					m_surface_textures[i]->bind(unit++);
			}
		}

		// Bind custom textures
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