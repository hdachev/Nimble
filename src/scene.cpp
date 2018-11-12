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

	Scene* Scene::load(const std::string& file)
	{
		std::string scene_json;
		
		if (!utility::read_text(file, scene_json))
			return nullptr;
		
		nlohmann::json json = nlohmann::json::parse(scene_json.c_str());

		Scene* scene = new Scene();
		
		std::string sceneName = json["name"];
		scene->m_name = sceneName;
		std::string envMap = json["environment_map"];
		TextureCube* cube = (TextureCube*)demo::load_image(envMap, GL_RGB16F, GL_RGB, GL_HALF_FLOAT);
		scene->m_env_map = cube;

		std::string irradianceMap = json["irradiance_map"];
		cube = (TextureCube*)demo::load_image(irradianceMap, GL_RGB16F, GL_RGB, GL_HALF_FLOAT);
		scene->m_irradiance_map = cube;
		scene->m_irradiance_map->set_min_filter(GL_LINEAR);
		
		std::string prefilteredMap = json["prefiltered_map"];
		cube = (TextureCube*)demo::load_image(prefilteredMap, GL_RGB16F, GL_RGB, GL_HALF_FLOAT);
		scene->m_prefiltered_map = cube;

		auto entities = json["entities"];

		for (auto& entity : entities)
		{
			Material* mat_override = nullptr;

			std::string name = entity["name"];
			std::string model = entity["mesh"];
			
			if (!entity["material"].is_null())
			{
				std::string material = entity["material"];
				mat_override = demo::load_material(material);
			}

			auto positionJson = entity["position"];
			glm::vec3 position = glm::vec3(positionJson[0], positionJson[1], positionJson[2]);

			auto scaleJson = entity["scale"];
			glm::vec3 scale = glm::vec3(scaleJson[0], scaleJson[1], scaleJson[2]);

			auto rotationJson = entity["rotation"];
			glm::vec3 rotation = glm::vec3(rotationJson[0], rotationJson[1], rotationJson[2]);

			Mesh* mesh = demo::load_mesh(model);
			Entity* new_entity = scene->create_entity();

			new_entity->m_override_mat = mat_override;
			new_entity->m_name = name;
			new_entity->m_position = position;
			new_entity->m_rotation = rotation;
			new_entity->m_scale = scale;
			new_entity->m_mesh = mesh;

			glm::mat4 H = glm::rotate(glm::mat4(1.0f), glm::radians(rotation.x), glm::vec3(0.0f, 1.0f, 0.0f));
			glm::mat4 P = glm::rotate(glm::mat4(1.0f), glm::radians(rotation.y), glm::vec3(1.0f, 0.0f, 0.0f));
			glm::mat4 B = glm::rotate(glm::mat4(1.0f), glm::radians(rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));

			glm::mat4 R = H * P * B;
			glm::mat4 S = glm::scale(glm::mat4(1.0f), scale);
			glm::mat4 T = glm::translate(glm::mat4(1.0f), position);

			new_entity->m_transform = T * R * S;
			new_entity->m_prev_transform = new_entity->m_transform;

			auto shaderJson = entity["shader"];

			Shader* shaders[2];

			std::vector<std::string> defines;

			if (mat_override)
			{
				if (mat_override->texture(TEXTURE_ALBEDO))
					defines.push_back("ALBEDO_TEXTURE");
				if (mat_override->texture(TEXTURE_NORMAL))
					defines.push_back("NORMAL_TEXTURE");
				if (mat_override->texture(TEXTURE_METALNESS))
					defines.push_back("METALNESS_TEXTURE");
				if (mat_override->texture(TEXTURE_ROUGHNESS))
					defines.push_back("ROUGHNESS_TEXTURE");
				if (mat_override->texture(TEXTURE_DISPLACEMENT))
					defines.push_back("HEIGHT_TEXTURE");
				if (mat_override->texture(TEXTURE_EMISSIVE))
					defines.push_back("EMISSIVE_TEXTURE");
			}
			else
			{
				defines.push_back("ALBEDO_TEXTURE");
				defines.push_back("NORMAL_TEXTURE");
				defines.push_back("METALNESS_TEXTURE");
				defines.push_back("ROUGHNESS_TEXTURE");
			}

			std::string vsFile = shaderJson["vs"];
			shaders[0] = GlobalGraphicsResources::load_shader(GL_VERTEX_SHADER, vsFile, defines);

			std::string fsFile = shaderJson["fs"];
			shaders[1] = GlobalGraphicsResources::load_shader(GL_FRAGMENT_SHADER, fsFile, defines);

			std::string combName = vsFile + fsFile;
			new_entity->m_program = GlobalGraphicsResources::load_program(combName, 2, &shaders[0]);
			
			new_entity->m_program->uniform_block_binding("u_PerFrame", 0);
			new_entity->m_program->uniform_block_binding("u_PerEntity", 1);
			new_entity->m_program->uniform_block_binding("u_PerScene", 2);
		}

		return scene;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	Scene::Scene() {}

	// -----------------------------------------------------------------------------------------------------------------------------------

	Scene::~Scene()
	{
		NIMBLE_SAFE_DELETE(m_env_map);
		NIMBLE_SAFE_DELETE(m_irradiance_map);
		NIMBLE_SAFE_DELETE(m_prefiltered_map);

		Entity** entities = m_entities.data();

		for (int i = 0; i < m_entities.size(); i++)
		{
			if (entities[i]->m_override_mat)
				Material::unload(entities[i]->m_override_mat);
			Mesh::unload(entities[i]->m_mesh);
		}
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

	void Scene::save(std::string path)
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------
}