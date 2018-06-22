#include "demo_loader.h"
#include <fstream>
#include <macros.h>
#include <utility.h>
#include <logger.h>
#include <material.h>
#include <json.hpp>

#define READ_AND_OFFSET(stream, dest, size, offset) stream.read((char*)dest, size); offset += size; stream.seekg(offset);

namespace demo
{
	// -----------------------------------------------------------------------------------------------------------------------------------
	// Structures - Image
	// -----------------------------------------------------------------------------------------------------------------------------------

	struct ImageHeader
	{
		uint8_t  compression;
		uint8_t  channel_size;
		uint8_t  num_channels;
		uint16_t num_array_slices;
		uint8_t  num_mip_slices;
	};

	struct MipSliceHeader
	{
		uint16_t width;
		uint16_t height;
		int size;
	};

	struct ImageFileHeader
	{
		uint32_t magic;
		uint8_t  version;
		uint8_t  type;
	};

	// -----------------------------------------------------------------------------------------------------------------------------------
	// Structures - Mesh
	// -----------------------------------------------------------------------------------------------------------------------------------

	struct MeshFileHeader
	{
		uint8_t   mesh_type;
		uint16_t  mesh_count;
		uint16_t  material_count;
		uint32_t  vertex_count;
		uint32_t  index_count;
		glm::vec3 max_extents;
		glm::vec3 min_extents;
		char 	name[50];
	};


	struct MeshHeader
	{
		uint8_t material_index;
		uint32_t index_count;
		uint32_t base_vertex;
		uint32_t base_index;
		glm::vec3  max_extents;
		glm::vec3  min_extents;
	};

	struct MeshMaterialJson
	{
		char material[50];
	};

	struct MeshMaterial
	{
		char albedo[50];
		char normal[50];
		char roughness[50];
		char metalness[50];
		char displacement[50];
	};


	// -----------------------------------------------------------------------------------------------------------------------------------

	dw::Texture* load_image(const std::string& path, GLenum internal_format, GLenum format, GLenum type)
	{
		dw::Texture* texture;

		std::fstream f(dw::utility::path_for_resource("assets/" + path), std::ios::in | std::ios::binary);

		if (!f.is_open())
		{
			DW_LOG_ERROR("Failed to open file: " + path);
			return nullptr;
		}

		ImageFileHeader file_header;
		uint16_t name_length = 0;
		char name[256];
		ImageHeader image_header;

		long offset = 0;

		f.seekp(offset);

		READ_AND_OFFSET(f, &file_header, sizeof(ImageFileHeader), offset);
		READ_AND_OFFSET(f, &name_length, sizeof(uint16_t), offset);
		READ_AND_OFFSET(f, &name[0], sizeof(char) * name_length, offset);

		name[name_length] = '\0';

#if defined(LOADER_PRINT_DEBUG_INFO)
		std::cout << "Name: " << name << std::endl;
#endif

		READ_AND_OFFSET(f, &image_header, sizeof(ImageHeader), offset);

#if defined(LOADER_PRINT_DEBUG_INFO)
		std::cout << "Channel Size: " << image_header.channel_size << std::endl;
		std::cout << "Channel Count: " << image_header.num_channels << std::endl;
		std::cout << "Array Slice Count: " << image_header.num_array_slices << std::endl;
		std::cout << "Mip Slice Count: " << image_header.num_mip_slices << std::endl;
#endif
		
		for (int array_slice = 0; array_slice < image_header.num_array_slices; array_slice++)
		{
#if defined(LOADER_PRINT_DEBUG_INFO)
			std::cout << std::endl;
			std::cout << "Array Slice: " << array_slice << std::endl;
#endif

			for (int mip_slice = 0; mip_slice < image_header.num_mip_slices; mip_slice++)
			{
				MipSliceHeader mip_header;
				char* image_data;

				READ_AND_OFFSET(f, &mip_header, sizeof(MipSliceHeader), offset);

				if (array_slice == 0 && mip_slice == 0)
				{
					if (image_header.num_array_slices == 6)
						texture = new dw::TextureCube(mip_header.width, mip_header.height, 1, image_header.num_mip_slices, internal_format, format, type);
					else
						texture = new dw::Texture2D(mip_header.width, mip_header.height, 1, image_header.num_mip_slices, 1, internal_format, format, type);
				}

#if defined(LOADER_PRINT_DEBUG_INFO)
				std::cout << std::endl;
				std::cout << "Mip Slice: " << mipSlice << std::endl;
				std::cout << "Width: " << mip_header.width << std::endl;
				std::cout << "Height: " << mip_header.height << std::endl;
#endif

				image_data = (char*)malloc(mip_header.size);
				READ_AND_OFFSET(f, image_data, mip_header.size, offset);

				if (image_header.num_array_slices == 6)
				{
					dw::TextureCube* cube = static_cast<dw::TextureCube*>(texture);
					cube->set_data(array_slice, 0, mip_slice, image_data);
				}
				else
				{
					dw::Texture2D* tex2D = static_cast<dw::Texture2D*>(texture);
					tex2D->set_data(array_slice, mip_slice, image_data);
				}

				free(image_data);
			}
		}

		f.close();

		return texture;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	dw::Mesh* load_mesh(const std::string& path)
	{
		if (dw::Mesh::is_loaded(path))
			return dw::Mesh::load(path);
		else
		{
			char* data = nullptr;
			FILE *file = fopen(dw::utility::path_for_resource("assets/" + path).c_str(), "rb");
			fseek(file, 0, SEEK_END);
			long len = ftell(file);
			data = (char*)malloc(len);
			rewind(file);
			fread(data, len, 1, file);

			size_t offset = 0;

			MeshFileHeader header;

			memcpy(&header, data, sizeof(MeshFileHeader));
			offset += sizeof(MeshFileHeader);

			int sub_mesh_count = header.mesh_count;
			dw::SubMesh* submeshes = new dw::SubMesh[sub_mesh_count];

			dw::Vertex* vertices = new dw::Vertex[header.vertex_count];
			uint32_t* indices = new uint32_t[header.index_count];
			MeshMaterialJson* mats = new MeshMaterialJson[header.material_count];
			MeshHeader* mesh_headers = new MeshHeader[header.mesh_count];

			memcpy(vertices, data + offset, sizeof(dw::Vertex) * header.vertex_count);
			offset += sizeof(dw::Vertex) * header.vertex_count;

			memcpy(indices, data + offset, sizeof(uint32_t) * header.index_count);
			offset += sizeof(uint32_t) * header.index_count;

			memcpy(mesh_headers, data + offset, sizeof(MeshHeader) * header.mesh_count);
			offset += sizeof(MeshHeader) * header.mesh_count;

			memcpy(mats, data + offset, sizeof(MeshMaterialJson) * header.material_count);

			for (uint32_t i = 0; i < header.mesh_count; i++)
			{
				submeshes[i].base_index = mesh_headers[i].base_index;
				submeshes[i].base_vertex = mesh_headers[i].base_vertex;
				submeshes[i].index_count = mesh_headers[i].index_count;
				submeshes[i].max_extents = mesh_headers[i].max_extents;
				submeshes[i].min_extents = mesh_headers[i].min_extents;

				if (header.material_count > 0 && mesh_headers[i].material_index < header.material_count)
				{
					std::string matName = mats[mesh_headers[i].material_index].material;

					if (!matName.empty() && matName != " ")
						submeshes[i].mat = load_material(matName);
					else
						submeshes[i].mat = nullptr;
				}
				else
					submeshes[i].mat = nullptr;
			}

			dw::Mesh* mesh = dw::Mesh::load(path, header.vertex_count, vertices, header.index_count, indices, header.mesh_count, submeshes, header.max_extents, header.min_extents);

			delete[] mats;
			delete[] mesh_headers;

			free(data);
			fclose(file);

			return mesh;
		}
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	dw::Material* load_material(const std::string& path)
	{
		if (dw::Material::is_loaded(path))
			return dw::Material::load(path, nullptr);
		else
		{
			std::string mat_json;
			bool result = dw::utility::read_text(dw::utility::path_for_resource("assets/" + path), mat_json);
			assert(result);
			nlohmann::json json = nlohmann::json::parse(mat_json.c_str());

			dw::Texture2D* textures[16];
			glm::vec4 albedo;
			float metalness;
			float roughness;
			int num_textures = 0;

			if (json.find("diffuse_map") != json.end())
			{
				std::string tex_path = json["diffuse_map"];
				textures[num_textures++] = dw::Material::load_texture(dw::utility::path_for_resource("assets/texture/" + tex_path), true);

				if (!textures[num_textures - 1])
				{
					DW_LOG_ERROR("Failed to load Albedo Map");
				}
			}
			else if (json.find("diffuse_value") != json.end())
			{
				albedo.x = json["diffuse_value"]["r"];
				albedo.y = json["diffuse_value"]["g"];
				albedo.z = json["diffuse_value"]["b"];
				albedo.w = json["diffuse_value"]["a"];
			}

			if (json.find("normal_map") != json.end())
			{
				std::string tex_path = json["normal_map"];
				textures[num_textures++] = dw::Material::load_texture(dw::utility::path_for_resource("assets/texture/" + tex_path));

				if (!textures[num_textures - 1])
				{
					DW_LOG_ERROR("Failed to load Normal Map");
				}
			}

			if (json.find("metalness_map") != json.end())
			{
				std::string tex_path = json["metalness_map"];
				textures[num_textures++] = dw::Material::load_texture(dw::utility::path_for_resource("assets/texture/" + tex_path));

				if (!textures[num_textures - 1])
				{
					DW_LOG_ERROR("Failed to load Metalness Map");
				}
			}
			if (json.find("metalness_value") != json.end())
			{
				metalness = json["metalness_value"];
			}

			if (json.find("roughness_map") != json.end())
			{
				std::string tex_path = json["roughness_map"];
				textures[num_textures++] = dw::Material::load_texture(dw::utility::path_for_resource("assets/texture/" + tex_path));

				if (!textures[num_textures - 1])
				{
					DW_LOG_ERROR("Failed to load Roughness Map");
				}
			}
			if (json.find("roughness_value") != json.end())
			{
				roughness = json["roughness_value"];
			}

			return dw::Material::load(path, num_textures, &textures[0], albedo, roughness, metalness);
		}
	}
	// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace demo