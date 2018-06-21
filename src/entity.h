#pragma once

#include <mesh.h>
#include <material.h>
#include "ogl.h"
#include <string>

struct Entity
{
	uint32_t id;
	std::string m_name;
	glm::vec3 m_position;
	glm::vec3 m_rotation;
	glm::vec3 m_scale;
	glm::mat4 m_transform;
	glm::mat4 m_prev_transform;
	dw::Program* m_program;
	dw::Material* m_override_mat;
	dw::Mesh* m_mesh;
};