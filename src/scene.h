#pragma once

#include "entity.h"
#include "packed_array.h"
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
		~Scene();

		// Updates entity transforms.
		void update();

		// Entity manipulation methods.
		Entity::ID create_entity(const std::string& name);
		Entity::ID lookup_entity_id(const std::string& name);
		Entity& lookup_entity(const std::string& name);
		Entity& lookup_entity(const Entity::ID& id);
		void update_entity(Entity e);
		void destroy_entity(const Entity::ID& id);
		void destroy_entity(const std::string& name);

		// Inline setters.
		inline void set_name(const std::string& name) { m_name = name; }
		inline void set_environment_map(const std::shared_ptr<TextureCube>& texture) { m_env_map = texture; }
		inline void set_irradiance_map(const std::shared_ptr<TextureCube>& texture) { m_irradiance_map = texture; }
		inline void set_prefiltered_map(const std::shared_ptr<TextureCube>& texture) { m_prefiltered_map = texture; }

		// Inline getters.
		inline uint32_t entity_count() { return m_entities.size(); }
		inline Entity* entities() { return &m_entities._objects[0]; }
		inline std::string name() const { return m_name; }
		inline std::shared_ptr<TextureCube>& env_map() { return m_env_map; }
		inline std::shared_ptr<TextureCube>& irradiance_map() { return m_irradiance_map; }
		inline std::shared_ptr<TextureCube>& prefiltered_map() { return m_prefiltered_map; }
		

	private:
		std::string						  m_name;
		PackedArray<ReflectionProbe, 128> m_reflection_probes;
		PackedArray<GIProbe, 128>		  m_gi_probes;
		PackedArray<Entity, 1024>		  m_entities;
		// PBR cubemaps common to the entire scene.
		std::shared_ptr<TextureCube>	  m_env_map;
		std::shared_ptr<TextureCube>	  m_irradiance_map;
		std::shared_ptr<TextureCube>	  m_prefiltered_map;
	};
}