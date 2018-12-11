#pragma once

#include "ogl.h"
#include "mesh.h"
#include "static_hash_map.h"
#include "shader_key.h"

namespace nimble
{
#define MAX_SHADER_CACHE_SIZE 10000
#define MAX_PROGRAM_CACHE_SIZE 10000

	class ShaderLibrary
	{
	public:
		ShaderLibrary(const std::string& vs, const std::string& fs);
		~ShaderLibrary();

		Program* lookup_program(const ProgramKey& key);
		Program* create_program(const MeshType& type, const uint32_t& flags, const std::shared_ptr<Material>& material);

	private:
		StaticHashMap<uint64_t, Program*, MAX_PROGRAM_CACHE_SIZE> m_program_cache;
		std::unordered_map<uint64_t, Shader*> m_vs_cache;
		std::unordered_map<uint64_t, Shader*> m_fs_cache;
		std::string m_vs_template;
		std::string m_fs_template;
	};
}