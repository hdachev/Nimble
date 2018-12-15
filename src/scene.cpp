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

	Scene::~Scene()
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	Entity::ID Scene::create_entity(const std::string& name)
	{
		Entity::ID id = m_entities.add();

		Entity& e = m_entities.lookup(id);

		e.m_id = id;
		e.m_name = name;

		return id;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	Entity::ID Scene::lookup_entity_id(const std::string& name)
	{
		for (uint32_t i = 0; i < m_entities.size(); i++)
		{
			if (m_entities._objects[i].m_name == name)
				return m_entities._objects[i].m_id;
		}

		return USHRT_MAX;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	Entity& Scene::lookup_entity(const std::string& name)
	{
		Entity::ID id = lookup_entity_id(name);
		return m_entities.lookup(id);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	Entity& Scene::lookup_entity(const Entity::ID& id)
	{
		return m_entities.lookup(id);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Scene::update_entity(Entity e)
	{
		Entity& old_entity = lookup_entity(e.m_id);
		old_entity = e;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Scene::destroy_entity(const Entity::ID& id)
	{
		if (m_entities.has(id))
			m_entities.remove(id);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Scene::destroy_entity(const std::string& name)
	{
		Entity::ID id = lookup_entity_id(name);

		if (id != USHRT_MAX && m_entities.has(id))
		{
			Entity& e = lookup_entity(id);
			e.~Entity();

			m_entities.remove(id);
		}
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	ReflectionProbe::ID Scene::create_reflection_probe(const glm::vec3& position, const glm::vec3& extents)
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	GIProbe::ID Scene::create_gi_probe(const glm::vec3& position)
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	ReflectionProbe& Scene::lookup_reflection_probe(const ReflectionProbe::ID& id)
	{
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	GIProbe& Scene::lookup_gi_probe(const GIProbe::ID& id)
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Scene::destroy_reflection_probe(const ReflectionProbe::ID& id)
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Scene::destroy_gi_probe(const GIProbe::ID& id)
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Scene::update_reflection_probes()
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Scene::update_gi_probes()
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Scene::update()
	{
		for (int i = 0; i < m_entities.size(); i++)
		{
			Entity& e = m_entities._objects[i];

			if (e.m_dirty)
			{
				glm::mat4 H = glm::rotate(glm::mat4(1.0f), glm::radians(e.m_rotation.x), glm::vec3(0.0f, 1.0f, 0.0f));
				glm::mat4 P = glm::rotate(glm::mat4(1.0f), glm::radians(e.m_rotation.y), glm::vec3(1.0f, 0.0f, 0.0f));
				glm::mat4 B = glm::rotate(glm::mat4(1.0f), glm::radians(e.m_rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));

				glm::mat4 R = H * P * B;
				glm::mat4 S = glm::scale(glm::mat4(1.0f), e.m_scale);
				glm::mat4 T = glm::translate(glm::mat4(1.0f), e.m_position);

				e.m_prev_transform = e.m_transform;
				e.m_transform = T * R * S;
			}
		}
	}

	// -----------------------------------------------------------------------------------------------------------------------------------
}