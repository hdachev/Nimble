#include "resource_manager.h"
#include "logger.h"
#include "ogl.h"
#include <runtime/loader.h>

namespace nimble
{
	static const GLenum kInternalFormatTable[][4] = 
	{
		{ GL_R8, GL_RG8, GL_RGB8, GL_RGBA8 },
		{ GL_R16F, GL_RG16F, GL_RGB16F, GL_RGBA16F },
		{ GL_R32F, GL_RG32F, GL_RGB32F, GL_RGBA32F }
	};

	static const GLenum kFormatTable[] =
	{
		GL_RED,
		GL_RG,
		GL_RGB,
		GL_RGBA
	};

	static const GLenum kTypeTable[] =
	{
		GL_UNSIGNED_BYTE,
		GL_HALF_FLOAT,
		GL_FLOAT
	};

	ResourceManager::ResourceManager()
	{

	}

	ResourceManager::~ResourceManager()
	{

	}

	Texture* ResourceManager::load_texture(const std::string& path)
	{
		Texture* texture = nullptr;

		if (m_texture_cache.find(path) != m_texture_cache.end())
			return m_texture_cache[path];
		else
		{
			ast::Image image;

			if (ast::load_image(path, image))
			{
				uint32_t type = 0;

				if (image.type == ast::PIXEL_TYPE_FLOAT16)
					type = 1;
				else if (image.type == ast::PIXEL_TYPE_FLOAT32)
					type = 2;

				// @TODO: Check for SRGB
				// @TODO: Accomodate Cubemaps
				// @TODO: Check for compressed formats
				texture = new Texture2D(image.data[0][0].width, image.data[0][0].height, image.array_slices, image.mip_slices, 1, kInternalFormatTable[type][image.components], kFormatTable[image.components], kTypeTable[type]);

				for (uint32_t i = 0; i < image.mip_slices; i++)
				{

				}
			}
			else
			{
				NIMBLE_LOG_ERROR("Failed to load Texture: " + path);
				return nullptr;
			}
		}
	}

	Material* ResourceManager::load_material(const std::string& path)
	{

	}

	Mesh* ResourceManager::load_mesh(const std::string& path)
	{

	}

	void ResourceManager::unload_texture(Texture* texture)
	{

	}

	void ResourceManager::unload_material(Material* material)
	{

	}

	void ResourceManager::unload_mesh(Mesh* mesh)
	{

	}
}