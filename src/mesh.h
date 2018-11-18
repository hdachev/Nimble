#pragma once

#include "material.h"
#include <common/mesh.h>
#include <memory>

namespace nimble
{
	class VertexBuffer;
	class IndexBuffer;
	class VertexArray;
	class Program;

	class Mesh
	{
	public:
		Mesh(const std::string& name,
			const glm::vec3& max_extents,
			const glm::vec3& min_extents,
			const std::vector<ast::SubMesh>& submeshes,
			const std::vector<std::shared_ptr<Material>>& materials,
			VertexBuffer* vertex_buffer,
			IndexBuffer* index_buffer,
			VertexArray* vertex_array);
		~Mesh();
		void bind();
		void bind_material(Program* program, const uint32_t& index);
		ast::SubMesh& submesh(const uint32_t& index);
		uint32_t submesh_count();

	private:
		std::string                 m_name;
		glm::vec3                   m_max_extents;
		glm::vec3                   m_min_extents;
		std::vector<ast::SubMesh>   m_submeshes;
		std::vector<std::shared_ptr<Material>>      m_materials;
		VertexBuffer*			    m_vertex_buffer;
		IndexBuffer*				m_index_buffer;
		VertexArray*				m_vertex_array;
	};
}