#include "scene.h"
#include <json.hpp>
#include <gtc/matrix_transform.hpp>
#include "material.h"
#include "macros.h"
#include "utility.h"
#include "demo_loader.h"
#include "global_graphics_resources.h"

namespace nimble
{
	// -----------------------------------------------------------------------------------------------------------------------------------

	Scene::Scene(const std::string& name) : m_name(name)
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	Scene::Scene(const std::string& name,
				 const std::vector<Entity*>& entities,
				 const std::vector<ReflectionProbe>& reflection_probes,
				 const std::vector<GIProbe>& gi_probes) : m_name(name), m_entities(entities), m_reflection_probes(reflection_probes), m_gi_probes(gi_probes)
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	Scene::~Scene()
	{
		Entity** entities = m_entities.data();

		for (int i = 0; i < m_entities.size(); i++)
			destroy_entity(m_entities[i]);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	Entity* Scene::create_entity()
	{
		Entity* e = new Entity();
		e->id = m_entities.size();
		m_entities.push_back(e);

		return e;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	Entity* Scene::lookup(const std::string& name)
	{
		for (int i = 0; i < m_entities.size(); i++)
		{
			Entity* e = m_entities[i];

			if (e->m_name == name)
				return e;
		}

		return nullptr;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Scene::destroy_entity(Entity* entity)
	{
		if (entity)
		{
			m_entities.erase(m_entities.begin() + entity->id);
			NIMBLE_SAFE_DELETE(entity);
		}
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Scene::update()
	{
		for (int i = 0; i < m_entities.size(); i++)
		{
			Entity* e = m_entities[i];

			glm::mat4 H = glm::rotate(glm::mat4(1.0f), glm::radians(e->m_rotation.x), glm::vec3(0.0f, 1.0f, 0.0f));
			glm::mat4 P = glm::rotate(glm::mat4(1.0f), glm::radians(e->m_rotation.y), glm::vec3(1.0f, 0.0f, 0.0f));
			glm::mat4 B = glm::rotate(glm::mat4(1.0f), glm::radians(e->m_rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));

			glm::mat4 R = H * P * B;
			glm::mat4 S = glm::scale(glm::mat4(1.0f), e->m_scale);
			glm::mat4 T = glm::translate(glm::mat4(1.0f), e->m_position);

			e->m_prev_transform = e->m_transform;
			e->m_transform = T * R * S;
		}
	}

	// -----------------------------------------------------------------------------------------------------------------------------------
}