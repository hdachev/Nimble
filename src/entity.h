#pragma once

#include <mesh.h>
#include <material.h>
#include "ogl.h"
#include <string>

namespace nimble
{
	struct Entity
	{
		uint32_t id;
		std::string m_name;
		glm::vec3 m_position;
		glm::vec3 m_rotation;
		glm::vec3 m_scale;
		glm::mat4 m_transform;
		glm::mat4 m_prev_transform;
		Program* m_program;
		Material* m_override_mat;
		Mesh* m_mesh;
	};
}