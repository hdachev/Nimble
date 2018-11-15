#include "resource_manager.h"
#include "logger.h"
#include <runtime/loader.h>

namespace nimble
{
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