#pragma once

#include <iostream>
#include <unordered_map>

namespace nimble
{
	class Texture;
	class Material;
	class Mesh;

	class ResourceManager
	{
	public:
		ResourceManager();
		~ResourceManager();

		Texture* load_texture(const std::string& path, const bool& srgb = false, const bool& cubemap = false);
		Material* load_material(const std::string& path);
		Mesh* load_mesh(const std::string& path);

		void unload_texture(Texture* texture);
		void unload_material(Material* material);
		void unload_mesh(Mesh* mesh);

	private:
		std::unordered_map<std::string, Texture*>  m_texture_cache;
		std::unordered_map<std::string, Material*> m_material_cache;
		std::unordered_map<std::string, Mesh*>	   m_mesh_cache;
	};
}