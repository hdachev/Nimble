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
		bool					  m_static;

#ifdef ENABLE_SUBMESH_CULLING
		std::vector<uint64_t>	  m_submesh_visibility;
#endif

		Entity()
		{
			m_static = false;
			m_dirty = true;
		}

		inline void set_position(const glm::vec3& position) { m_position = position; m_dirty = true; }
		inline void set_rotation(const glm::vec3& rotation) { m_rotation = rotation; m_dirty = true; }
		inline void set_scale(const glm::vec3& scale) { m_scale = scale; m_dirty = true; }
		inline bool visibility(const uint32_t& view_index) { return m_visibility & BIT_FLAG(view_index) == 1; }
		inline void set_visible(const uint32_t& view_index) {  SET_BIT(m_visibility, view_index); }
		inline void set_invisible(const uint32_t& view_index) { CLEAR_BIT(m_visibility, view_index); }

#ifdef ENABLE_SUBMESH_CULLING
		inline bool submesh_visibility(const uint32_t& submesh_index, const uint32_t& view_index) { return m_submesh_visibility[submesh_index] & BIT_FLAG(view_index) == 1; }
		inline void set_submesh_visible(const uint32_t& submesh_index, const uint32_t& view_index) { SET_BIT(m_submesh_visibility[submesh_index], view_index); }
		inline void set_submesh_invisible(const uint32_t& submesh_index, const uint32_t& view_index) { CLEAR_BIT(m_submesh_visibility[submesh_index], view_index); }
#endif
	};
}