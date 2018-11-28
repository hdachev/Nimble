#pragma once

#include "ogl.h"
#include "mesh.h"
#include "static_hash_map.h"

namespace nimble
{
#define MAX_SHADER_CACHE_SIZE 10000
#define MAX_PROGRAM_CACHE_SIZE 10000

	// 16-bits  | 3-bits	
	// Material | Mesh Type 

	struct ShaderKey
	{
		uint64_t key;

		inline void set_material_id(const uint32_t& value) { uint64_t temp = value; key |= temp; }
		inline void set_mesh_type(const uint32_t& value) { uint64_t temp = value; key |= (temp << 16); }

		inline uint32_t material_id() { return key & 0xffff; }
		inline uint32_t mesh_type() { return (key >> 16) & 7; }
	};

	class ShaderLibrary
	{
	public:
		ShaderLibrary(const std::string& vs, const std::string& fs);
		~ShaderLibrary();

		Program* lookup_program(const MeshType& type, const std::shared_ptr<Material>& material);

	private:
		StaticHashMap<uint64_t, std::shared_ptr<Program>, MAX_PROGRAM_CACHE_SIZE> m_program_cache;
		std::string m_vs_template;
		std::string m_fs_template;
	};
}