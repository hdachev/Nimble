#include "render_node.h"
#include "render_graph.h"
#include "profiler.h"
#include "view.h"
#include "scene.h"
#include "shader_library.h"
#include "logger.h"
#include "renderer.h"

namespace nimble
{
// -----------------------------------------------------------------------------------------------------------------------------------

RenderNode::RenderNode(RenderGraph* graph) :
    m_enabled(true), m_graph(graph), m_total_time_cpu(0.0f), m_total_time_gpu(0.0f)
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

RenderNode::~RenderNode()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::shared_ptr<RenderTarget> RenderNode::find_output_render_target(const std::string& name)
{
    for (auto& output : m_output_rts)
    {
        if (output.slot_name == name)
            return output.render_target;
    }

    return nullptr;
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::shared_ptr<RenderTarget> RenderNode::find_intermediate_render_target(const std::string& name)
{
    for (auto& rt : m_intermediate_rts)
    {
        if (rt.first == name)
            return rt.second;
    }

    return nullptr;
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::shared_ptr<RenderTarget> RenderNode::find_input_render_target(const std::string& name)
{
    InputRenderTarget* rt = find_input_render_target_slot(name);

    if (rt)
        return rt->prev_render_target;

    return nullptr;
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::shared_ptr<ShaderStorageBuffer> RenderNode::find_output_buffer(const std::string& name)
{
    OutputBuffer* buffer = find_output_buffer_slot(name);

    if (buffer)
        return buffer->buffer;

    return nullptr;
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::shared_ptr<ShaderStorageBuffer> RenderNode::find_input_buffer(const std::string& name)
{
    InputBuffer* buffer = find_input_buffer_slot(name);

    if (buffer)
        return buffer->prev_buffer;

    return nullptr;
}

// -----------------------------------------------------------------------------------------------------------------------------------

RenderNode::OutputRenderTarget* RenderNode::find_output_render_target_slot(const std::string& name)
{
    for (auto& output : m_output_rts)
    {
        if (output.slot_name == name)
            return &output;
    }

    return nullptr;
}

// -----------------------------------------------------------------------------------------------------------------------------------

RenderNode::InputRenderTarget* RenderNode::find_input_render_target_slot(const std::string& name)
{
    for (auto& input : m_input_rts)
    {
        if (input.slot_name == name)
            return &input;
    }

    return nullptr;
}

// -----------------------------------------------------------------------------------------------------------------------------------

RenderNode::OutputBuffer* RenderNode::find_output_buffer_slot(const std::string& name)
{
    for (auto& output : m_output_buffers)
    {
        if (output.slot_name == name)
            return &output;
    }

    return nullptr;
}

// -----------------------------------------------------------------------------------------------------------------------------------

RenderNode::InputBuffer* RenderNode::find_input_buffer_slot(const std::string& name)
{
    for (auto& input : m_input_buffers)
    {
        if (input.slot_name == name)
            return &input;
    }

    return nullptr;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RenderNode::set_input(const std::string& name, OutputRenderTarget* rt, std::shared_ptr<RenderNode> owner)
{
    for (auto& input : m_input_rts)
    {
        if (input.slot_name == name)
        {
            input.prev_slot_name     = rt->slot_name;
            input.prev_render_target = rt->render_target;
            input.prev_node          = owner;
            return;
        }
    }

    NIMBLE_LOG_ERROR("No input render target slot named: " + name);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RenderNode::set_input(const std::string& name, OutputBuffer* buffer, std::shared_ptr<RenderNode> owner)
{
    for (auto& input : m_input_buffers)
    {
        if (input.slot_name == name)
        {
            input.prev_slot_name = buffer->slot_name;
            input.prev_buffer    = buffer->buffer;
            input.prev_node      = owner;
            return;
        }
    }

    NIMBLE_LOG_ERROR("No input buffer slot named: " + name);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RenderNode::timing_total(float& cpu_time, float& gpu_time)
{
    cpu_time = m_total_time_cpu;
    gpu_time = m_total_time_gpu;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RenderNode::set_bool_parameter(const std::string& name, bool value)
{
    for (auto& param : m_bool_parameters)
    {
        if (name == param.name && param.ptr)
        {
            *param.ptr = value;
            return;
        }
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RenderNode::set_int_parameter(const std::string& name, int32_t value)
{
    for (auto& param : m_int_parameters)
    {
        if (name == param.name && param.ptr)
        {
            *param.ptr = value;
            return;
        }
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RenderNode::set_float_parameter(const std::string& name, float value)
{
    for (auto& param : m_float_parameters)
    {
        if (name == param.name && param.ptr)
        {
            *param.ptr = value;
            return;
        }
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

BoolParameter* RenderNode::bool_parameters(int32_t& count)
{
    count = m_bool_parameters.size();
    return &m_bool_parameters[0];
}

// -----------------------------------------------------------------------------------------------------------------------------------

IntParameter* RenderNode::int_parameters(int32_t& count)
{
    count = m_int_parameters.size();
    return &m_int_parameters[0];
}

// -----------------------------------------------------------------------------------------------------------------------------------

FloatParameter* RenderNode::float_parameters(int32_t& count)
{
    count = m_float_parameters.size();
    return &m_float_parameters[0];
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RenderNode::register_bool_parameter(const std::string& name, bool& parameter)
{
    m_bool_parameters.push_back({ &parameter, name });
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RenderNode::register_int_parameter(const std::string& name, int32_t& parameter)
{
    m_int_parameters.push_back({ &parameter, name });
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RenderNode::register_float_parameter(const std::string& name, float& parameter)
{
    m_float_parameters.push_back({ &parameter, name });
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RenderNode::bind_shadow_maps(Renderer* renderer, Program* program, int32_t tex_unit, uint32_t flags)
{
    if (program)
    {
        if ((HAS_BIT_FLAG(flags, NODE_USAGE_SHADOW_MAPPING)) && program->set_uniform("s_DirectionalLightShadowMaps", tex_unit))
            renderer->directional_light_shadow_maps()->bind(tex_unit++);

        if ((HAS_BIT_FLAG(flags, NODE_USAGE_SHADOW_MAPPING)) && program->set_uniform("s_SpotLightShadowMaps", tex_unit))
            renderer->spot_light_shadow_maps()->bind(tex_unit++);

        if ((HAS_BIT_FLAG(flags, NODE_USAGE_SHADOW_MAPPING)) && program->set_uniform("s_PointLightShadowMaps", tex_unit))
            renderer->point_light_shadow_maps()->bind(tex_unit++);
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RenderNode::declare_connections()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RenderNode::on_window_resized(const uint32_t& w, const uint32_t& h)
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RenderNode::trigger_cascade_view_render(View* view)
{
    m_graph->trigger_cascade_view_render(view);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RenderNode::register_input_render_target(const std::string& name)
{
    for (auto& input : m_input_rts)
    {
        if (input.slot_name == name)
        {
            NIMBLE_LOG_ERROR("Input render target slot already registered: " + name);
            return;
        }
    }

    m_input_rts.push_back({ name, "", nullptr, nullptr });
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RenderNode::register_input_buffer(const std::string& name)
{
    for (auto& input : m_input_buffers)
    {
        if (input.slot_name == name)
        {
            NIMBLE_LOG_ERROR("Input buffer slot already registered: " + name);
            return;
        }
    }

    m_input_buffers.push_back({ name, "", nullptr, nullptr });
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::shared_ptr<RenderTarget> RenderNode::register_output_render_target(const std::string& name, const uint32_t& w, const uint32_t& h, GLenum target, GLenum internal_format, GLenum format, GLenum type, uint32_t num_samples, uint32_t array_size, uint32_t mip_levels)
{
    for (auto& output : m_output_rts)
    {
        if (output.slot_name == name)
        {
            NIMBLE_LOG_ERROR("Output render target already registered: " + name);
            return nullptr;
        }
    }

    std::shared_ptr<RenderTarget> rt = std::make_shared<RenderTarget>();

    rt->w               = w;
    rt->h               = h;
    rt->scale_w         = 0.0f;
    rt->scale_h         = 0.0f;
    rt->target          = target;
    rt->internal_format = internal_format;
    rt->format          = format;
    rt->type            = type;
    rt->num_samples     = num_samples;
    rt->array_size      = array_size;
    rt->mip_levels      = mip_levels;

    m_output_rts.push_back({ name, rt });

    return rt;
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::shared_ptr<RenderTarget> RenderNode::register_forwarded_output_render_target(const std::string& input)
{
    for (auto& output : m_output_rts)
    {
        if (output.slot_name == input)
        {
            NIMBLE_LOG_ERROR("Output render target already registered: " + input);
            return nullptr;
        }
    }

    std::shared_ptr<RenderTarget> rt = std::make_shared<RenderTarget>();

    rt->forward_slot = input;

    m_output_rts.push_back({ input, rt });

    return rt;
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::shared_ptr<RenderTarget> RenderNode::register_scaled_output_render_target(const std::string& name, const float& w, const float& h, GLenum target, GLenum internal_format, GLenum format, GLenum type, uint32_t num_samples, uint32_t array_size, uint32_t mip_levels)
{
    for (auto& output : m_output_rts)
    {
        if (output.slot_name == name)
        {
            NIMBLE_LOG_ERROR("Output render target already registered: " + name);
            return nullptr;
        }
    }

    std::shared_ptr<RenderTarget> rt = std::make_shared<RenderTarget>();

    rt->w               = 0;
    rt->h               = 0;
    rt->scale_w         = w;
    rt->scale_h         = h;
    rt->target          = target;
    rt->internal_format = internal_format;
    rt->format          = format;
    rt->type            = type;
    rt->num_samples     = num_samples;
    rt->array_size      = array_size;
    rt->mip_levels      = mip_levels;

    m_output_rts.push_back({ name, rt });

    return rt;
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::shared_ptr<RenderTarget> RenderNode::register_intermediate_render_target(const std::string& name, const uint32_t& w, const uint32_t& h, GLenum target, GLenum internal_format, GLenum format, GLenum type, uint32_t num_samples, uint32_t array_size, uint32_t mip_levels)
{
    for (auto& output : m_intermediate_rts)
    {
        if (output.first == name)
        {
            NIMBLE_LOG_ERROR("Intermediate render target already registered: " + name);
            return nullptr;
        }
    }

    std::shared_ptr<RenderTarget> rt = std::make_shared<RenderTarget>();

    rt->w               = w;
    rt->h               = h;
    rt->scale_w         = 0.0f;
    rt->scale_h         = 0.0f;
    rt->target          = target;
    rt->internal_format = internal_format;
    rt->format          = format;
    rt->type            = type;
    rt->num_samples     = num_samples;
    rt->array_size      = array_size;
    rt->mip_levels      = mip_levels;

    m_intermediate_rts.push_back({ name, rt });

    return rt;
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::shared_ptr<RenderTarget> RenderNode::register_scaled_intermediate_render_target(const std::string& name, const float& w, const float& h, GLenum target, GLenum internal_format, GLenum format, GLenum type, uint32_t num_samples, uint32_t array_size, uint32_t mip_levels)
{
    for (auto& output : m_intermediate_rts)
    {
        if (output.first == name)
        {
            NIMBLE_LOG_ERROR("Intermediate render target already registered: " + name);
            return nullptr;
        }
    }

    std::shared_ptr<RenderTarget> rt = std::make_shared<RenderTarget>();

    rt->w               = 0;
    rt->h               = 0;
    rt->scale_w         = w;
    rt->scale_h         = h;
    rt->target          = target;
    rt->internal_format = internal_format;
    rt->format          = format;
    rt->type            = type;
    rt->num_samples     = num_samples;
    rt->array_size      = array_size;
    rt->mip_levels      = mip_levels;

    m_intermediate_rts.push_back({ name, rt });

    return rt;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RenderNode::render_scene(Renderer* renderer, Scene* scene, View* view, ShaderLibrary* library, uint32_t flags, std::function<void(View*, Program*, int32_t&)> function)
{
    if (scene)
    {
        Entity* entities = scene->entities();

        // Bind buffers
        if (HAS_BIT_FLAG(flags, NODE_USAGE_PER_VIEW_UBO))
            renderer->per_view_ssbo()->bind_range(0, sizeof(PerViewUniforms) * view->uniform_idx, sizeof(PerViewUniforms));

        if (HAS_BIT_FLAG(flags, NODE_USAGE_POINT_LIGHTS) || HAS_BIT_FLAG(flags, NODE_USAGE_SPOT_LIGHTS) || HAS_BIT_FLAG(flags, NODE_USAGE_DIRECTIONAL_LIGHTS))
            renderer->per_scene_ssbo()->bind_base(2);

        for (uint32_t i = 0; i < scene->entity_count(); i++)
        {
            Entity& e = entities[i];

            if (!view->culling || (view->culling && e.visibility(view->cull_idx)))
            {
                // Bind mesh VAO
                e.mesh->bind();

                for (uint32_t j = 0; j < e.mesh->submesh_count(); j++)
                {
                    SubMesh& s = e.mesh->submesh(j);

                    int32_t tex_unit = 0;

#ifdef ENABLE_SUBMESH_CULLING
                    if (!view->culling || (view->culling && e.submesh_visibility(j, view->cull_idx)))
                    {
#endif
                        ProgramKey& key = s.material->program_key();

                        key.set_mesh_type(e.mesh->type());

                        // Lookup shader program from library
                        Program* program = library->lookup_program(key);

                        if (!program)
                        {
                            program = library->create_program(e.mesh->type(),
                                                              flags,
                                                              s.material,
                                                              m_graph->type() == RENDER_GRAPH_STANDARD ? renderer->directional_light_render_graph() : nullptr,
                                                              m_graph->type() == RENDER_GRAPH_STANDARD ? renderer->spot_light_render_graph() : nullptr,
                                                              m_graph->type() == RENDER_GRAPH_STANDARD ? renderer->point_light_render_graph() : nullptr);
                        }

						program->use();

                        // Bind material
                        if (HAS_BIT_FLAG(flags, NODE_USAGE_MATERIAL_ALBEDO) && !s.material->surface_texture(TEXTURE_TYPE_ALBEDO))
                            program->set_uniform("u_Albedo", s.material->uniform_albedo());

                        if (HAS_BIT_FLAG(flags, NODE_USAGE_MATERIAL_EMISSIVE) && !s.material->surface_texture(TEXTURE_TYPE_EMISSIVE))
                            program->set_uniform("u_Emissive", s.material->uniform_emissive());

                        if ((HAS_BIT_FLAG(flags, NODE_USAGE_MATERIAL_ROUGH_SMOOTH) && !s.material->surface_texture(TEXTURE_TYPE_ROUGH_SMOOTH)) || (HAS_BIT_FLAG(flags, NODE_USAGE_MATERIAL_METAL_SPEC) && !s.material->surface_texture(TEXTURE_TYPE_METAL_SPEC)))
                            program->set_uniform("u_MetalRough", glm::vec4(s.material->uniform_metallic(), s.material->uniform_roughness(), 0.0f, 0.0f));

                        s.material->bind(program, tex_unit);

                        bind_shadow_maps(renderer, program, tex_unit, flags);

                        if (HAS_BIT_FLAG(flags, NODE_USAGE_PER_OBJECT_UBO))
                            renderer->per_entity_ubo()->bind_range(1, sizeof(PerEntityUniforms) * i, sizeof(PerEntityUniforms));

                        if (function)
                            function(view, program, tex_unit);

                        glDrawElementsBaseVertex(GL_TRIANGLES, s.index_count, GL_UNSIGNED_INT, (void*)(sizeof(unsigned int) * s.base_index), s.base_vertex);

#ifdef ENABLE_SUBMESH_CULLING
                    }
#endif
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RenderNode::render_fullscreen_triangle(Renderer* renderer, View* view, Program* program, int32_t tex_unit, uint32_t flags)
{
    // Bind buffers
    if (HAS_BIT_FLAG(flags, NODE_USAGE_PER_VIEW_UBO))
        renderer->per_view_ssbo()->bind_range(0, sizeof(PerViewUniforms) * view->uniform_idx, sizeof(PerViewUniforms));

    if (HAS_BIT_FLAG(flags, NODE_USAGE_POINT_LIGHTS) || HAS_BIT_FLAG(flags, NODE_USAGE_SPOT_LIGHTS) || HAS_BIT_FLAG(flags, NODE_USAGE_DIRECTIONAL_LIGHTS))
        renderer->per_scene_ssbo()->bind_base(1);

    bind_shadow_maps(renderer, program, tex_unit, flags);

    // Render fullscreen triangle
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RenderNode::render_fullscreen_quad(Renderer* renderer, View* view, Program* program, int32_t tex_unit, uint32_t flags)
{
    // Bind buffers
    if (HAS_BIT_FLAG(flags, NODE_USAGE_PER_VIEW_UBO))
        renderer->per_view_ssbo()->bind_range(0, sizeof(PerViewUniforms) * view->uniform_idx, sizeof(PerViewUniforms));

    if (HAS_BIT_FLAG(flags, NODE_USAGE_POINT_LIGHTS) || HAS_BIT_FLAG(flags, NODE_USAGE_SPOT_LIGHTS) || HAS_BIT_FLAG(flags, NODE_USAGE_DIRECTIONAL_LIGHTS))
        renderer->per_scene_ssbo()->bind_base(1);

    bind_shadow_maps(renderer, program, tex_unit, flags);

    // Render fullscreen triangle
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}

// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble