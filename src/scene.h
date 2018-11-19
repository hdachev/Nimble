#pragma once

#include "entity.h"
#include <vector>
#include <string>

namespace nimble
{
	struct ReflectionProbe
	{
		std::shared_ptr<Texture> texture;
		glm::vec3 extents;
		glm::vec3 position;
	};

	struct GIProbe
	{
		std::shared_ptr<Texture> texture;
		glm::vec3 position;
	};

	class Scene
	{
	public:
		Scene(const std::string& name);
		Scene(const std::string& name,
			  const std::vector<Entity*>& entities, 
			  const std::vector<ReflectionProbe>& reflection_probes, 
			  const std::vector<GIProbe>& gi_probes);
		~Scene();

		// Updates entity transforms.
		void update();

		// Create a new uninitialized entity.
		Entity* create_entity();

		// Find a created entity by name.
		Entity* lookup(const std::string& name);

		// Destroy an entity by passing in a pointer. Won't destroy the associated resources for now.
		void destroy_entity(Entity* entity);

		// Inline getters.
		inline uint32_t entity_count() { return m_entities.size(); }
		inline Entity** entities() { return m_entities.data(); }
		inline const char* name() { return m_name.c_str(); }
		inline std::shared_ptr<TextureCube>& env_map() { return m_env_map; }
		inline std::shared_ptr<TextureCube>& irradiance_map() { return m_irradiance_map; }
		inline std::shared_ptr<TextureCube>& prefiltered_map() { return m_prefiltered_map; }
		

	private:
		// PBR cubemaps common to the entire scene.
		std::string					 m_name;
		std::shared_ptr<TextureCube> m_env_map;
		std::shared_ptr<TextureCube> m_irradiance_map;
		std::shared_ptr<TextureCube> m_prefiltered_map;
		std::vector<Entity*>		 m_entities;
		std::vector<ReflectionProbe> m_reflection_probes;
		std::vector<GIProbe>		 m_gi_probes;
	};
}