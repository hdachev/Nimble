#pragma once

#include "mesh.h"
#include "material.h"
#include "ogl.h"
#include <string>

namespace nimble
{
	struct Entity
	{
		using ID = uint32_t;

		ID						  m_id;
		std::string				  m_name;
		glm::vec3				  m_position;
		glm::vec3				  m_rotation;
		glm::vec3				  m_scale;
		glm::mat4				  m_transform;
		glm::mat4				  m_prev_transform;
		std::shared_ptr<Material> m_override_mat;
		std::shared_ptr<Mesh>	  m_mesh;
		uint64_t				  m_visibility;
		bool					  m_dirty;

#ifdef ENABLE_SUBMESH_CULLING
		std::vector<uint64_t>	  m_submesh_visibility;
#endif

		Entity()
		{
			m_dirty = true;
		}

		inline void set_position(const glm::vec3& position) { m_position = position; m_dirty = true; }
		inline void set_rotation(const glm::vec3& rotation) { m_rotation = rotation; m_dirty = true; }
		inline void set_scale(const glm::vec3& scale) { m_scale = scale; m_dirty = true; }
	};
}