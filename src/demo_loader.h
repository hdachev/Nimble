#pragma once

#include <string>
#include <ogl.h>
#include <mesh.h>

enum DemoTextureTypes
{
	TEXTURE_ALBEDO = 0,
	TEXTURE_NORMAL = 1,
	TEXTURE_METALNESS = 2,
	TEXTURE_ROUGHNESS = 3,
	TEXTURE_DISPLACEMENT = 4,
	TEXTURE_EMISSIVE = 5
};

// Free functions for loading custom versions Textures and Meshes.
namespace demo
{
	extern dw::Texture* load_image(const std::string& path, GLenum internal_format, GLenum format, GLenum type);
	extern dw::Mesh* load_mesh(const std::string& path);
	extern dw::Material* load_material(const std::string& path);
} // namespace demo