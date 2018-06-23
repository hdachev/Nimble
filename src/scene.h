#pragma once

#include "entity.h"
#include <vector>
#include <string>

class Scene
{
public:
	// Creates a scene from a description stored in a JSON file.
	static Scene* load(const std::string& file);

	Scene();
	~Scene();

	// Updates entity transforms.
	void update();

	// Create a new uninitialized entity.
	Entity* create_entity();

	// Find a created entity by name.
	Entity* lookup(const std::string& name);

	// Destroy an entity by passing in a pointer. Won't destroy the associated resources for now.
	void destroy_entity(Entity* entity);

	// TODO: Save scene description to JSON.
	void save(std::string path);

	// Inline getters.
	inline uint32_t entity_count() { return m_entities.size(); }
	inline Entity** entities() { return m_entities.data(); }
	inline const char* name() { return m_name.c_str(); }
	inline dw::TextureCube* env_map() { return m_env_map; }
	inline dw::TextureCube* irradiance_map() { return m_irradiance_map; }
	inline dw::TextureCube* prefiltered_map() { return m_prefiltered_map; }
	

private:
	// PBR cubemaps common to the entire scene.
	dw::TextureCube*     m_env_map;
	dw::TextureCube*	 m_irradiance_map;
	dw::TextureCube*     m_prefiltered_map;
	std::string			 m_name;
	std::vector<Entity*> m_entities;
};