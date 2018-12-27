#pragma once

#include "geometry.h"
#include "material.h"
#include <common/mesh.h>
#include <memory>

namespace nimble
{
	class VertexBuffer;
	class IndexBuffer;
	class VertexArray;
	class Program;

	enum MeshType
	{
		MESH_TYPE_STATIC = 0,
		MESH_TYPE_SKELETAL
	};

	struct SubMesh
	{
		uint32_t  index_count;
		uint32_t  base_vertex;
		uint32_t  base_index;
		glm::vec3 max_extents;
		glm::vec3 min_extents;
		std::shared_ptr<Material> material;
	};

	class Mesh
	{
	public:
		Mesh(const std::string& name,
			const glm::vec3& max_extents,
			const glm::vec3& min_extents,
			const std::vector<SubMesh>& submeshes,
			std::shared_ptr<VertexBuffer> vertex_buffer,
			std::shared_ptr<IndexBuffer> index_buffer,
			std::shared_ptr<VertexArray> vertex_array);
		~Mesh();
		void bind();
		SubMesh& submesh(const uint32_t& index);
		uint32_t submesh_count();
		AABB aabb();

		// Inline getters
		inline MeshType type() { return m_type; }

	private:
		MeshType					m_type = MESH_TYPE_STATIC;
		std::string                 m_name;
		AABB						m_aabb;
		std::vector<SubMesh>   m_submeshes;
		std::shared_ptr<VertexBuffer> m_vertex_buffer;
		std::shared_ptr<IndexBuffer>  m_index_buffer;
		std::shared_ptr<VertexArray>  m_vertex_array;
	};
}