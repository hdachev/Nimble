#pragma once

#include <iostream>
#include <unordered_map>
#include <memory>

namespace nimble
{
	class Texture;
	class Material;
	class Mesh;
	class Scene;
	class Shader;
	class Program;

	class ResourceManager
	{
	public:
		ResourceManager();
		~ResourceManager();

		std::shared_ptr<Texture> load_texture(const std::string& path, const bool& absolute = false, const bool& srgb = false, const bool& cubemap = false);
		std::shared_ptr<Material> load_material(const std::string& path, const bool& absolute = false);
		std::shared_ptr<Mesh> load_mesh(const std::string& path);
		std::shared_ptr<Scene> load_scene(const std::string& path);
		std::shared_ptr<Shader> load_shader(const std::string& path, const uint32_t& type, std::vector<std::string> defines = std::vector<std::string>());
		
	private:
		bool read_shader(const std::string& path, std::string& out, const std::vector<std::string>& defines);

	private:
		std::unordered_map<std::string, std::weak_ptr<Texture>>  m_texture_cache;
		std::unordered_map<std::string, std::weak_ptr<Material>> m_material_cache;
		std::unordered_map<std::string, std::weak_ptr<Mesh>>	 m_mesh_cache;
		std::unordered_map<std::string, std::weak_ptr<Scene>>	 m_scene_cache;
		std::unordered_map<std::string, std::weak_ptr<Shader>>	 m_shader_cache;
		std::unordered_map<std::string, uint32_t>				 m_vertex_func_id_map;
		std::unordered_map<std::string, uint32_t>				 m_fragment_func_id_map;
	};
}