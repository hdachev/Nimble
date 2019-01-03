#include "scene.h"
#include <json.hpp>
#include <gtc/matrix_transform.hpp>
#include "material.h"
#include "macros.h"
#include "utility.h"
#include "global_graphics_resources.h"

namespace nimble
{
	// -----------------------------------------------------------------------------------------------------------------------------------

	Scene::Scene(const std::string& name) : m_name(name)
	{
		m_camera = std::make_unique<Camera>(60.0f, 0.1f, 1000.0f, 16.0f/9.0f, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f));
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

		e.id = id;
		e.name = name;

		return id;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	Entity::ID Scene::lookup_entity_id(const std::string& name)
	{
		for (uint32_t i = 0; i < m_entities.size(); i++)
		{
			if (m_entities._objects[i].name == name)
				return m_entities._objects[i].id;
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
		Entity& old_entity = lookup_entity(e.id);
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
		ReflectionProbe::ID id = m_reflection_probes.add();

		ReflectionProbe& p = m_reflection_probes.lookup(id);

		p.id = id;
		p.position = position;
		p.extents = extents;

		return id;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	GIProbe::ID Scene::create_gi_probe(const glm::vec3& position)
	{
		GIProbe::ID id = m_gi_probes.add();

		GIProbe& p = m_gi_probes.lookup(id);

		p.id = id;
		p.position = position;

		return id;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	ReflectionProbe& Scene::lookup_reflection_probe(const ReflectionProbe::ID& id)
	{
		return m_reflection_probes.lookup(id);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	GIProbe& Scene::lookup_gi_probe(const GIProbe::ID& id)
	{
		return m_gi_probes.lookup(id);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Scene::destroy_reflection_probe(const ReflectionProbe::ID& id)
	{
		if (m_reflection_probes.has(id))
			m_reflection_probes.remove(id);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Scene::destroy_gi_probe(const GIProbe::ID& id)
	{
		if (m_gi_probes.has(id))
			m_gi_probes.remove(id);
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

			if (e.dirty)
				e.transform.update();
		}
	}

	// -----------------------------------------------------------------------------------------------------------------------------------
}