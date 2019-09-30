#pragma once

#include "glad.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <glm.hpp>

//#define NIMBLE_ENABLE_GL_ERROR_CHECK
// OpenGL error checking macro.
#ifdef NIMBLE_ENABLE_GL_ERROR_CHECK
#    define GL_CHECK_ERROR(x)                                                                              \
        x;                                                                                                 \
        {                                                                                                  \
            GLenum err(glGetError());                                                                      \
                                                                                                           \
            while (err != GL_NO_ERROR)                                                                     \
            {                                                                                              \
                std::string error;                                                                         \
                                                                                                           \
                switch (err)                                                                               \
                {                                                                                          \
                    case GL_INVALID_OPERATION: error = "INVALID_OPERATION"; break;                         \
                    case GL_INVALID_ENUM: error = "INVALID_ENUM"; break;                                   \
                    case GL_INVALID_VALUE: error = "INVALID_VALUE"; break;                                 \
                    case GL_OUT_OF_MEMORY: error = "OUT_OF_MEMORY"; break;                                 \
                    case GL_INVALID_FRAMEBUFFER_OPERATION: error = "INVALID_FRAMEBUFFER_OPERATION"; break; \
                }                                                                                          \
                                                                                                           \
                std::string formatted_error = "OPENGL: ";                                                  \
                formatted_error             = formatted_error + error;                                     \
                NIMBLE_LOG_ERROR(formatted_error);                                                         \
                err = glGetError();                                                                        \
            }                                                                                              \
        }
#else
#    define GL_CHECK_ERROR(x) x
#endif

namespace nimble
{
extern GLenum format_from_internal_format(GLenum fmt);
extern GLenum type_from_internal_format(GLenum fmt);

// Texture base class.
class Texture
{
public:
    Texture();
    virtual ~Texture();

    // Bind texture to specified texture unit i.e GL_TEXTURE<unit>.
    void bind(uint32_t unit);
    void unbind(uint32_t unit);

    // Binding to image units.
    void bind_image(uint32_t unit, uint32_t mip_level, uint32_t layer, GLenum access, GLenum format);

    // Mipmap generation.
    void generate_mipmaps();

    // Getters.
    GLuint   id();
    GLenum   target();
    uint32_t array_size();
    uint32_t version();
    uint32_t mip_levels();

    // Texture sampler functions.
    void set_wrapping(GLenum s, GLenum t, GLenum r);
    void set_border_color(float r, float g, float b, float a);
    void set_min_filter(GLenum filter);
    void set_mag_filter(GLenum filter);
    void set_compare_mode(GLenum mode);
    void set_compare_func(GLenum func);

    inline GLenum internal_format() { return m_internal_format; }
    inline GLenum format() { return m_format; }

protected:
    GLuint   m_gl_tex = UINT32_MAX;
    GLenum   m_target;
    GLenum   m_internal_format;
    GLenum   m_format;
    GLenum   m_type;
    uint32_t m_version = 0;
    uint32_t m_array_size;
    uint32_t m_mip_levels;
};

#if !defined(__EMSCRIPTEN__)
class Texture1D : public Texture
{
public:
    Texture1D(uint32_t w, uint32_t array_size, int32_t mip_levels, GLenum internal_format, GLenum format, GLenum type);
    ~Texture1D();
    void     set_data(int array_index, int mip_level, void* data);
    uint32_t width();

private:
    uint32_t m_width;
};
#endif

class Texture2D : public Texture
{
public:
    Texture2D(uint32_t w, uint32_t h, uint32_t array_size, int32_t mip_levels, uint32_t num_samples, GLenum internal_format, GLenum format, GLenum type, bool compressed = false);
    ~Texture2D();
    void     set_data(int array_index, int mip_level, void* data);
    void     set_compressed_data(int array_index, int mip_level, size_t size, void* data);
    void     data(int mip_level, int array_index, void* data);
    void     extents(int mip_level, int& width, int& height);
    void     resize(uint32_t w, uint32_t h);
    uint32_t width();
    uint32_t height();
    uint32_t num_samples();

protected:
    Texture2D();

    bool     m_compressed;
    uint32_t m_width;
    uint32_t m_height;
    uint32_t m_num_samples;
};

class Texture3D : public Texture
{
public:
    Texture3D(uint32_t w, uint32_t h, uint32_t d, int mip_levels, GLenum internal_format, GLenum format, GLenum type);
    ~Texture3D();
    void     set_data(int mip_level, void* data);
    void     data(int mip_level, void* data);
    void     extents(int mip_level, int& width, int& height, int& depth);
    uint32_t width();
    uint32_t height();
    uint32_t depth();

private:
    uint32_t m_width;
    uint32_t m_height;
    uint32_t m_depth;
};

class TextureCube : public Texture
{
public:
    TextureCube(uint32_t w, uint32_t h, uint32_t array_size, int32_t mip_levels, GLenum internal_format, GLenum format, GLenum type, bool compressed = false);
    ~TextureCube();
    void     set_data(int face_index, int layer_index, int mip_level, void* data);
    void     set_compressed_data(int face_index, int layer_index, int mip_level, size_t size, void* data);
    uint32_t width();
    uint32_t height();

private:
    uint32_t m_width;
    uint32_t m_height;
};

class Texture2DView : public Texture2D
{
public:
    Texture2DView(TextureCube* origin_tex, uint32_t min_level, uint32_t num_levels, uint32_t face);
    Texture2DView(Texture2D* origin_tex, uint32_t min_level, uint32_t num_levels, uint32_t layer, uint32_t num_layers);
    ~Texture2DView();
};

class Framebuffer
{
public:
    Framebuffer();
    ~Framebuffer();
    void bind();
    void unbind();

    // Attach entire texture or entire layer of a layered texture as a render target.
    void attach_render_target(uint32_t attachment, Texture* texture, uint32_t layer, uint32_t mip_level, bool draw = true, bool read = true);

    // Attach multiple render targets.
    void attach_multiple_render_targets(uint32_t attachment_count, Texture** texture);

    // Attach a given face from a cubemap or a specific layer of a cubemap array as a render target.
    void attach_render_target(uint32_t attachment, TextureCube* texture, uint32_t face, uint32_t layer, uint32_t mip_level, bool draw = true, bool read = true);

    // Attach entire texture or entire layer of a layered texture as a depth stencil target.
    void attach_depth_stencil_target(Texture* texture, uint32_t layer, uint32_t mip_level);

    // Attach a given face from a cubemap or a specific layer of a cubemap array as a depth stencil target.
    void attach_depth_stencil_target(TextureCube* texture, uint32_t face, uint32_t layer, uint32_t mip_level);

private:
    void check_status();

private:
    GLuint m_gl_fbo;
};

class Shader
{
    friend class Program;

public:
    static Shader* create_from_file(GLenum type, std::string path);

    Shader(GLenum type, std::string source);
    ~Shader();
    GLenum type();
    bool   compiled();
    GLuint id();

private:
    bool   m_compiled;
    GLuint m_gl_shader;
    GLenum m_type;
};

class Program
{
public:
    Program(uint32_t count, Shader** shaders);
    ~Program();
    void    use();
    int32_t num_active_uniform_blocks();
    void    uniform_block_binding(std::string name, int binding);
    bool    set_uniform(std::string name, int value);
    bool    set_uniform(std::string name, float value);
    bool    set_uniform(std::string name, glm::vec2 value);
    bool    set_uniform(std::string name, glm::vec3 value);
    bool    set_uniform(std::string name, glm::vec4 value);
    bool    set_uniform(std::string name, glm::mat2 value);
    bool    set_uniform(std::string name, glm::mat3 value);
    bool    set_uniform(std::string name, glm::mat4 value);
    bool    set_uniform(std::string name, int count, int* value);
    bool    set_uniform(std::string name, int count, float* value);
    bool    set_uniform(std::string name, int count, glm::vec2* value);
    bool    set_uniform(std::string name, int count, glm::vec3* value);
    bool    set_uniform(std::string name, int count, glm::vec4* value);
    bool    set_uniform(std::string name, int count, glm::mat2* value);
    bool    set_uniform(std::string name, int count, glm::mat3* value);
    bool    set_uniform(std::string name, int count, glm::mat4* value);
    GLint   id();

private:
    GLuint                                  m_gl_program;
    int32_t                                 m_num_active_uniform_blocks;
    std::unordered_map<std::string, GLuint> m_location_map;
};

class Buffer
{
public:
    Buffer(GLenum type, GLenum usage, size_t size, void* data);
    virtual ~Buffer();
    void  bind();
    void  bind_base(int index);
    void  bind_range(int index, size_t offset, size_t size);
    void  unbind();
    void* map(GLenum access);
    void* map_range(GLenum access, size_t offset, size_t size);
    void  unmap();
    void  set_data(size_t offset, size_t size, void* data);
    void  flush_mapped_range(size_t offset, size_t length);

protected:
    GLenum m_type;
    GLuint m_gl_buffer;
    size_t m_size;
#if defined(__EMSCRIPTEN__)
    void*  m_staging;
    size_t m_mapped_size;
    size_t m_mapped_offset;
#endif
};

class VertexBuffer : public Buffer
{
public:
    VertexBuffer(GLenum usage, size_t size, void* data = nullptr);
    ~VertexBuffer();
};

class IndexBuffer : public Buffer
{
public:
    IndexBuffer(GLenum usage, size_t size, void* data = nullptr);
    ~IndexBuffer();
};

class UniformBuffer : public Buffer
{
public:
    UniformBuffer(GLenum usage, size_t size, void* data = nullptr);
    ~UniformBuffer();
};

#if !defined(__EMSCRIPTEN__)
class ShaderStorageBuffer : public Buffer
{
public:
    ShaderStorageBuffer(GLenum usage, size_t size, void* data = nullptr);
    ~ShaderStorageBuffer();
};
#endif

struct VertexAttrib
{
    uint32_t num_sub_elements;
    uint32_t type;
    bool     normalized;
    uint32_t offset;
};

class VertexArray
{
public:
    VertexArray(VertexBuffer* vbo, IndexBuffer* ibo, size_t vertex_size, int attrib_count, VertexAttrib attribs[]);
    ~VertexArray();
    void bind();
    void unbind();

private:
    GLuint m_gl_vao;
};

class Query
{
public:
    Query();
    ~Query();
    void query_counter(GLenum type);
    void begin(GLenum type);
    void end(GLenum type);
    void result_64(uint64_t* ptr);
    bool result_available();

private:
    GLuint m_query;
};

class Fence
{
public:
    Fence();
    ~Fence();
    void insert();
    void wait();

private: 
	GLsync m_fence = nullptr;
};

} // namespace nimble
