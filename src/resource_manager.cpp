#include "resource_manager.h"
#include "logger.h"
#include "ogl.h"
#include "material.h"
#include <runtime/loader.h>

namespace nimble
{
	static const GLenum kInternalFormatTable[][4] = 
	{
		{ GL_R8, GL_RG8, GL_RGB8, GL_RGBA8 },
		{ GL_R16F, GL_RG16F, GL_RGB16F, GL_RGBA16F },
		{ GL_R32F, GL_RG32F, GL_RGB32F, GL_RGBA32F }
	};

	static const GLenum kCompressedTable[][2] =
	{
		{ GL_COMPRESSED_RGB_S3TC_DXT1_EXT, GL_COMPRESSED_SRGB_S3TC_DXT1_EXT }, // BC1
		{ GL_COMPRESSED_RGBA_S3TC_DXT1_EXT, GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT }, // BC1a
		{ GL_COMPRESSED_RGBA_S3TC_DXT3_EXT, GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT }, // BC2
		{ GL_COMPRESSED_RGBA_S3TC_DXT5_EXT, GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT }, // BC3
		{ GL_COMPRESSED_RGBA_S3TC_DXT5_EXT, GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT }, // BC3n
		{ GL_COMPRESSED_RED_RGTC1, 0 }, // BC4
		{ GL_COMPRESSED_RG_RGTC2, 0 }, // BC5
		{ GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT_ARB, 0 }, // BC6
		{ GL_COMPRESSED_RGBA_BPTC_UNORM_ARB, GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM_ARB } // BC7
	};

	static const GLenum kFormatTable[] =
	{
		GL_RED,
		GL_RG,
		GL_RGB,
		GL_RGBA
	};

	static const GLenum kTypeTable[] =
	{
		GL_UNSIGNED_BYTE,
		GL_HALF_FLOAT,
		GL_FLOAT
	};

	ResourceManager::ResourceManager()
	{

	}

	ResourceManager::~ResourceManager()
	{

	}

	Texture* ResourceManager::load_texture(const std::string& path, const bool& srgb, const bool& cubemap)
	{
		if (m_texture_cache.find(path) != m_texture_cache.end())
			return m_texture_cache[path];
		else
		{
			ast::Image image;

			if (ast::load_image(path, image))
			{
				uint32_t type = 0;

				if (image.type == ast::PIXEL_TYPE_FLOAT16)
					type = 1;
				else if (image.type == ast::PIXEL_TYPE_FLOAT32)
					type = 2;

				if (cubemap)
				{
					if (image.array_slices != 6)
					{
						NIMBLE_LOG_ERROR("Texture does not have 6 array slices: " + path);
						return nullptr;
					}

					if (image.compression == ast::COMPRESSION_NONE)
					{
						TextureCube* texture = new TextureCube(image.data[0][0].width,
							image.data[0][0].height,
							image.array_slices,
							image.mip_slices,
							kInternalFormatTable[type][image.components - 1],
							kFormatTable[image.components - 1],
							kTypeTable[type]);

						for (uint32_t i = 0; i < image.array_slices; i++)
						{
							for (uint32_t j = 0; j < image.mip_slices; j++)
								texture->set_data(i, 0, j, image.data[i][j].data);
						}

						return texture;
					}
					else
					{
						if (kCompressedTable[image.compression - 1][(int)srgb] == 0)
						{
							NIMBLE_LOG_ERROR("No SRGB format available for this compression type: " + path);
							return nullptr;
						}

						TextureCube* texture = new TextureCube(image.data[0][0].width,
							image.data[0][0].height,
							1,
							image.mip_slices,
							kCompressedTable[image.compression - 1][(int)srgb],
							kFormatTable[image.components - 1],
							kTypeTable[type],
							true);

						for (uint32_t i = 0; i < image.array_slices; i++)
						{
							for (uint32_t j = 0; j < image.mip_slices; j++)
								texture->set_compressed_data(i, 0, j, image.data[i][j].size, image.data[i][j].data);
						}

						return texture;
					}
				}
				else
				{
					if (image.compression == ast::COMPRESSION_NONE)
					{
						Texture2D* texture = new Texture2D(image.data[0][0].width,
							image.data[0][0].height,
							image.array_slices,
							image.mip_slices,
							1,
							kInternalFormatTable[type][image.components - 1],
							kFormatTable[image.components - 1],
							kTypeTable[type]);

						for (uint32_t i = 0; i < image.array_slices; i++)
						{
							for (uint32_t j = 0; j < image.mip_slices; j++)
								texture->set_data(i, j, image.data[i][j].data);
						}

						return texture;
					}
					else
					{
						Texture2D* texture = new Texture2D(image.data[0][0].width,
							image.data[0][0].height,
							image.array_slices,
							image.mip_slices,
							1,
							kCompressedTable[image.compression - 1][(int)srgb],
							kFormatTable[image.components - 1],
							kTypeTable[type],
							true);

						for (uint32_t i = 0; i < image.array_slices; i++)
						{
							for (uint32_t j = 0; j < image.mip_slices; j++)
								texture->set_compressed_data(i, j, image.data[i][j].size, image.data[i][j].data);
						}

						return texture;
					}
				}
			}
			else
			{
				NIMBLE_LOG_ERROR("Failed to load Texture: " + path);
				return nullptr;
			}
		}
	}

	Material* ResourceManager::load_material(const std::string& path)
	{
		if (m_material_cache.find(path) != m_material_cache.end())
			return m_material_cache[path];
		else
		{
			ast::Material ast_material;

			if (ast::load_material(path, ast_material))
			{
				Material* material = new Material();



				return material;
			}
			else
			{
				NIMBLE_LOG_ERROR("Failed to load Material: " + path);
				return nullptr;
			}
		}
	}

	Mesh* ResourceManager::load_mesh(const std::string& path)
	{

	}

	void ResourceManager::unload_texture(Texture* texture)
	{

	}

	void ResourceManager::unload_material(Material* material)
	{

	}

	void ResourceManager::unload_mesh(Mesh* mesh)
	{

	}
}