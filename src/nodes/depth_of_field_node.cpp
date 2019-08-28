#include "depth_of_field_node.h"
#include "../render_graph.h"
#include "../resource_manager.h"
#include "../renderer.h"
#include "../profiler.h"

namespace nimble
{
DEFINE_RENDER_NODE_FACTORY(DepthOfFieldNode)

// -----------------------------------------------------------------------------------------------------------------------------------

DepthOfFieldNode::DepthOfFieldNode(RenderGraph* graph) :
    RenderNode(graph)
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

DepthOfFieldNode::~DepthOfFieldNode()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DepthOfFieldNode::declare_connections()
{
    register_input_render_target("Color");
    register_input_render_target("Depth");

    m_coc_rt              = register_scaled_intermediate_render_target("CoC", 1.0f, 1.0f, GL_TEXTURE_2D, GL_RG8, GL_RG, GL_UNSIGNED_BYTE);
    m_color4_rt           = register_scaled_intermediate_render_target("Color4", 0.5f, 0.5f, GL_TEXTURE_2D, GL_RGB16F, GL_RGB, GL_HALF_FLOAT);
    m_mul_coc_far4_rt     = register_scaled_intermediate_render_target("MulCoCFar4", 0.5f, 0.5f, GL_TEXTURE_2D, GL_RGB16F, GL_RGB, GL_HALF_FLOAT);
    m_coc4_rt             = register_scaled_intermediate_render_target("CoC4", 0.5f, 0.5f, GL_TEXTURE_2D, GL_RG8, GL_RG, GL_UNSIGNED_BYTE);
    m_near_coc_max_x4_rt  = register_scaled_intermediate_render_target("NearCoCMaxX4", 0.5f, 0.5f, GL_TEXTURE_2D, GL_R8, GL_RED, GL_UNSIGNED_BYTE);
    m_near_coc_max4_rt    = register_scaled_intermediate_render_target("NearCoCMax4", 0.5f, 0.5f, GL_TEXTURE_2D, GL_R8, GL_RED, GL_UNSIGNED_BYTE);
    m_near_coc_blur_x4_rt = register_scaled_intermediate_render_target("NearCoCBlurX4", 0.5f, 0.5f, GL_TEXTURE_2D, GL_R8, GL_RED, GL_UNSIGNED_BYTE);
    m_near_coc_blur4_rt   = register_scaled_intermediate_render_target("NearCoCBlur4", 0.5f, 0.5f, GL_TEXTURE_2D, GL_R8, GL_RED, GL_UNSIGNED_BYTE);
    m_near_dof4_rt        = register_scaled_intermediate_render_target("NearDoF4", 0.5f, 0.5f, GL_TEXTURE_2D, GL_RGB16F, GL_RGB, GL_HALF_FLOAT);
    m_far_dof4_rt         = register_scaled_intermediate_render_target("FarDoF4", 0.5f, 0.5f, GL_TEXTURE_2D, GL_RGB16F, GL_RGB, GL_HALF_FLOAT);
    m_near_fill_dof4_rt   = register_scaled_intermediate_render_target("NearFillDoF4", 0.5f, 0.5f, GL_TEXTURE_2D, GL_RGB16F, GL_RGB, GL_HALF_FLOAT);
    m_far_fill_dof4_rt    = register_scaled_intermediate_render_target("FarFillDoF4", 0.5f, 0.5f, GL_TEXTURE_2D, GL_RGB16F, GL_RGB, GL_HALF_FLOAT);
    m_composite_rt        = register_scaled_output_render_target("DoFComposite", 1.0f, 1.0f, GL_TEXTURE_2D, GL_RGB16F, GL_RGB, GL_HALF_FLOAT);
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool DepthOfFieldNode::initialize_private(Renderer* renderer, ResourceManager* res_mgr)
{
    register_bool_parameter("Enabled", m_enabled);

    m_color_rt = find_input_render_target("Color");
    m_depth_rt = find_input_render_target("Depth");

    m_coc_rt->texture->set_min_filter(GL_NEAREST);
    m_near_coc_max_x4_rt->texture->set_min_filter(GL_NEAREST);
    m_near_coc_max4_rt->texture->set_min_filter(GL_NEAREST);
    m_near_coc_blur_x4_rt->texture->set_min_filter(GL_NEAREST);
    m_near_coc_blur4_rt->texture->set_min_filter(GL_NEAREST);

    m_coc_rtv              = RenderTargetView(0, 0, 0, m_coc_rt->texture);
    m_color4_rtv           = RenderTargetView(0, 0, 0, m_color4_rt->texture);
    m_mul_coc_far4_rtv     = RenderTargetView(0, 0, 0, m_mul_coc_far4_rt->texture);
    m_coc4_rtv             = RenderTargetView(0, 0, 0, m_coc4_rt->texture);
    m_near_coc_max_x4_rtv  = RenderTargetView(0, 0, 0, m_near_coc_max_x4_rt->texture);
    m_near_coc_max4_rtv    = RenderTargetView(0, 0, 0, m_near_coc_max4_rt->texture);
    m_near_coc_blur_x4_rtv = RenderTargetView(0, 0, 0, m_near_coc_blur_x4_rt->texture);
    m_near_coc_blur4_rtv   = RenderTargetView(0, 0, 0, m_near_coc_blur4_rt->texture);
    m_near_dof4_rtv        = RenderTargetView(0, 0, 0, m_near_dof4_rt->texture);
    m_far_dof4_rtv         = RenderTargetView(0, 0, 0, m_far_dof4_rt->texture);
    m_near_fill_dof4_rtv   = RenderTargetView(0, 0, 0, m_near_fill_dof4_rt->texture);
    m_far_fill_dof4_rtv    = RenderTargetView(0, 0, 0, m_far_fill_dof4_rt->texture);
    m_composite_rtv        = RenderTargetView(0, 0, 0, m_composite_rt->texture);

    m_triangle_vs         = res_mgr->load_shader("shader/post_process/fullscreen_triangle_vs.glsl", GL_VERTEX_SHADER);
    m_coc_fs              = res_mgr->load_shader("shader/post_process/depth_of_field/coc_fs.glsl", GL_FRAGMENT_SHADER);
    m_downsample_fs       = res_mgr->load_shader("shader/post_process/depth_of_field/downsample_fs.glsl", GL_FRAGMENT_SHADER);
    m_near_coc_max_x4_fs  = res_mgr->load_shader("shader/post_process/filters_fs.glsl", GL_FRAGMENT_SHADER, { "HORIZONTAL", "MAX13", "CHANNELS_COUNT_1" });
    m_near_coc_max4_fs    = res_mgr->load_shader("shader/post_process/filters_fs.glsl", GL_FRAGMENT_SHADER, { "VERTICAL", "MAX13", "CHANNELS_COUNT_1" });
    m_near_coc_blur_x4_fs = res_mgr->load_shader("shader/post_process/filters_fs.glsl", GL_FRAGMENT_SHADER, { "HORIZONTAL", "BLUR13", "CHANNELS_COUNT_1" });
    m_near_coc_blur4_fs   = res_mgr->load_shader("shader/post_process/filters_fs.glsl", GL_FRAGMENT_SHADER, { "VERTICAL", "BLUR13", "CHANNELS_COUNT_1" });
    m_computation_fs      = res_mgr->load_shader("shader/post_process/depth_of_field/computation_fs.glsl", GL_FRAGMENT_SHADER);
    m_fill_fs             = res_mgr->load_shader("shader/post_process/depth_of_field/fill_fs.glsl", GL_FRAGMENT_SHADER);
    m_composite_fs        = res_mgr->load_shader("shader/post_process/depth_of_field/composite_fs.glsl", GL_FRAGMENT_SHADER);

    if (m_triangle_vs)
    {
        if (m_coc_fs)
            m_coc_program = renderer->create_program(m_triangle_vs, m_coc_fs);
        else
            return false;

        if (m_downsample_fs)
            m_downsample_program = renderer->create_program(m_triangle_vs, m_downsample_fs);
        else
            return false;

        if (m_near_coc_max_x4_fs)
            m_near_coc_max_x_program = renderer->create_program(m_triangle_vs, m_near_coc_max_x4_fs);
        else
            return false;

        if (m_near_coc_max4_fs)
            m_near_coc_max_program = renderer->create_program(m_triangle_vs, m_near_coc_max4_fs);
        else
            return false;

        if (m_near_coc_blur_x4_fs)
            m_near_coc_blur_x_program = renderer->create_program(m_triangle_vs, m_near_coc_blur_x4_fs);
        else
            return false;

        if (m_near_coc_blur4_fs)
            m_near_coc_blur_program = renderer->create_program(m_triangle_vs, m_near_coc_blur4_fs);
        else
            return false;

        if (m_computation_fs)
            m_computation_program = renderer->create_program(m_triangle_vs, m_computation_fs);
        else
            return false;

        if (m_fill_fs)
            m_fill_program = renderer->create_program(m_triangle_vs, m_fill_fs);
        else
            return false;

        if (m_composite_fs)
            m_composite_program = renderer->create_program(m_triangle_vs, m_composite_fs);
        else
            return false;

        return true;
    }
    else
        return false;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DepthOfFieldNode::execute(double delta, Renderer* renderer, Scene* scene, View* view)
{
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    if (m_enabled)
    {
        coc_generation(delta, renderer, scene, view);
        downsample(delta, renderer, scene, view);
        near_coc_max(delta, renderer, scene, view);
        near_coc_blur(delta, renderer, scene, view);
        dof_computation(delta, renderer, scene, view);
        fill(delta, renderer, scene, view);
        composite(delta, renderer, scene, view);
    }
    else
        blit_render_target(renderer, m_color_rt, m_composite_rt);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DepthOfFieldNode::shutdown()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::string DepthOfFieldNode::name()
{
    return "Depth Of Field";
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DepthOfFieldNode::coc_generation(double delta, Renderer* renderer, Scene* scene, View* view)
{
    NIMBLE_SCOPED_SAMPLE("CoC Generation");

    glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "CoC Generation");

    renderer->bind_render_targets(1, &m_coc_rtv, nullptr);

    glViewport(0, 0, m_graph->window_width(), m_graph->window_height());
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    m_coc_program->use();

    if (m_coc_program->set_uniform("s_Depth", 0))
        m_depth_rt->texture->bind(0);

    glm::vec2 pixel_size = glm::vec2(1.0f / float(m_graph->window_width()), 1.0f / float(m_graph->window_height()));
    m_coc_program->set_uniform("u_PixelSize", pixel_size);

    std::shared_ptr<Camera> camera = scene->camera();
    m_coc_program->set_uniform("u_NearBegin", camera->m_near_begin);
    m_coc_program->set_uniform("u_NearEnd", camera->m_near_end);
    m_coc_program->set_uniform("u_FarBegin", camera->m_far_begin);
    m_coc_program->set_uniform("u_FarEnd", camera->m_far_end);

    int32_t tex_unit = 0;
    render_fullscreen_triangle(renderer, view, m_coc_program.get(), tex_unit, NODE_USAGE_PER_VIEW_UBO);

    glPopDebugGroup();
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DepthOfFieldNode::downsample(double delta, Renderer* renderer, Scene* scene, View* view)
{
    NIMBLE_SCOPED_SAMPLE("Downsample");

    glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "Downsample");

    RenderTargetView rtvs[] = { m_color4_rtv, m_mul_coc_far4_rtv, m_coc4_rtv };
    renderer->bind_render_targets(3, rtvs, nullptr);

    glViewport(0, 0, m_graph->window_width() * 0.5f, m_graph->window_height() * 0.5f);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    m_downsample_program->use();

    glm::vec2 pixel_size = glm::vec2(1.0f / float(m_graph->window_width()), 1.0f / float(m_graph->window_height()));
    m_downsample_program->set_uniform("u_PixelSize", pixel_size);

    if (m_downsample_program->set_uniform("s_Color", 0))
        m_color_rt->texture->bind(0);

    if (m_downsample_program->set_uniform("s_CoC", 1))
        m_coc_rt->texture->bind(1);

    int32_t tex_unit = 0;
    render_fullscreen_triangle(renderer, view, m_downsample_program.get(), tex_unit, NODE_USAGE_PER_VIEW_UBO);

    glPopDebugGroup();
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DepthOfFieldNode::near_coc_max(double delta, Renderer* renderer, Scene* scene, View* view)
{
    NIMBLE_SCOPED_SAMPLE("Near CoC Max");

    glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "Near CoC Max");

    // Horizontal
    renderer->bind_render_targets(1, &m_near_coc_max_x4_rtv, nullptr);

    glViewport(0, 0, m_graph->window_width() * 0.5f, m_graph->window_height() * 0.5f);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    m_near_coc_max_x_program->use();

    glm::vec2 pixel_size = glm::vec2(1.0f / float(m_graph->window_width() / 2), 1.0f / float(m_graph->window_height() / 2));
    m_near_coc_max_x_program->set_uniform("u_PixelSize", pixel_size);

    if (m_near_coc_max_x_program->set_uniform("s_Texture", 0))
        m_coc4_rt->texture->bind(0);

    int32_t tex_unit = 0;
    render_fullscreen_triangle(renderer, view, m_near_coc_max_x_program.get(), tex_unit, NODE_USAGE_PER_VIEW_UBO);

    // Vertical
    renderer->bind_render_targets(1, &m_near_coc_max4_rtv, nullptr);

    glViewport(0, 0, m_graph->window_width() * 0.5f, m_graph->window_height() * 0.5f);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    m_near_coc_max_program->use();

    m_near_coc_max_program->set_uniform("u_PixelSize", pixel_size);

    if (m_near_coc_max_program->set_uniform("s_Texture", 0))
        m_near_coc_max_x4_rt->texture->bind(0);

    render_fullscreen_triangle(renderer, view, m_near_coc_max_program.get(), tex_unit, NODE_USAGE_PER_VIEW_UBO);

    glPopDebugGroup();
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DepthOfFieldNode::near_coc_blur(double delta, Renderer* renderer, Scene* scene, View* view)
{
    NIMBLE_SCOPED_SAMPLE("Near CoC Blur");

    glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "Near CoC Blur");

    // Horizontal
    renderer->bind_render_targets(1, &m_near_coc_blur_x4_rtv, nullptr);

    glViewport(0, 0, m_graph->window_width() * 0.5f, m_graph->window_height() * 0.5f);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    m_near_coc_blur_x_program->use();

    glm::vec2 pixel_size = glm::vec2(1.0f / float(m_graph->window_width() / 2), 1.0f / float(m_graph->window_height() / 2));
    m_near_coc_blur_x_program->set_uniform("u_PixelSize", pixel_size);

    if (m_near_coc_blur_x_program->set_uniform("s_Texture", 0))
        m_near_coc_max4_rt->texture->bind(0);

    int32_t tex_unit = 0;
    render_fullscreen_triangle(renderer, view, m_near_coc_blur_x_program.get(), tex_unit, NODE_USAGE_PER_VIEW_UBO);

    // Vertical
    renderer->bind_render_targets(1, &m_near_coc_blur4_rtv, nullptr);

    glViewport(0, 0, m_graph->window_width() * 0.5f, m_graph->window_height() * 0.5f);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    m_near_coc_blur_program->use();

    m_near_coc_blur_program->set_uniform("u_PixelSize", pixel_size);

    if (m_near_coc_blur_program->set_uniform("s_Texture", 0))
        m_near_coc_blur_x4_rt->texture->bind(0);

    render_fullscreen_triangle(renderer, view, m_near_coc_blur_program.get(), tex_unit, NODE_USAGE_PER_VIEW_UBO);

    glPopDebugGroup();
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DepthOfFieldNode::dof_computation(double delta, Renderer* renderer, Scene* scene, View* view)
{
    NIMBLE_SCOPED_SAMPLE("DoF Computation");

    glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "DoF Computation");

    RenderTargetView rtvs[] = { m_near_dof4_rtv, m_far_dof4_rtv };
    renderer->bind_render_targets(2, rtvs, nullptr);

    glViewport(0, 0, m_graph->window_width() * 0.5f, m_graph->window_height() * 0.5f);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    m_computation_program->use();

    glm::vec2 pixel_size = glm::vec2(1.0f / float(m_graph->window_width() / 2), 1.0f / float(m_graph->window_height() / 2));
    m_computation_program->set_uniform("u_PixelSize", pixel_size);
    m_computation_program->set_uniform("u_KernelSize", m_kernel_size);

    if (m_computation_program->set_uniform("s_CoC4", 0))
        m_coc4_rt->texture->bind(0);

    if (m_computation_program->set_uniform("s_NearBlurCoC4", 1))
        m_near_coc_blur4_rt->texture->bind(1);

    if (m_computation_program->set_uniform("s_Color4", 2))
        m_color4_rt->texture->bind(2);

    if (m_computation_program->set_uniform("s_ColorFarCoC4", 3))
        m_mul_coc_far4_rt->texture->bind(3);

    int32_t tex_unit = 0;
    render_fullscreen_triangle(renderer, view, m_computation_program.get(), tex_unit, NODE_USAGE_PER_VIEW_UBO);

    glPopDebugGroup();
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DepthOfFieldNode::fill(double delta, Renderer* renderer, Scene* scene, View* view)
{
    NIMBLE_SCOPED_SAMPLE("Fill");

    glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "Fill");

    RenderTargetView rtvs[] = { m_near_fill_dof4_rtv, m_far_fill_dof4_rtv };
    renderer->bind_render_targets(2, rtvs, nullptr);

    glViewport(0, 0, m_graph->window_width() * 0.5f, m_graph->window_height() * 0.5f);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    m_fill_program->use();

    glm::vec2 pixel_size = glm::vec2(1.0f / float(m_graph->window_width() / 2), 1.0f / float(m_graph->window_height() / 2));
    m_fill_program->set_uniform("u_PixelSize", pixel_size);

    if (m_fill_program->set_uniform("s_CoC4", 0))
        m_coc4_rt->texture->bind(0);

    if (m_fill_program->set_uniform("s_NearBlurCoC4", 1))
        m_near_coc_blur4_rt->texture->bind(1);

    if (m_fill_program->set_uniform("s_NearDoF4", 2))
        m_near_dof4_rt->texture->bind(2);

    if (m_fill_program->set_uniform("s_FarDoF4", 3))
        m_far_dof4_rt->texture->bind(3);

    int32_t tex_unit = 0;
    render_fullscreen_triangle(renderer, view, m_fill_program.get(), tex_unit, NODE_USAGE_PER_VIEW_UBO);

    glPopDebugGroup();
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DepthOfFieldNode::composite(double delta, Renderer* renderer, Scene* scene, View* view)
{
    NIMBLE_SCOPED_SAMPLE("Composite");

    glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "Composite");

    renderer->bind_render_targets(1, &m_composite_rtv, nullptr);

    glViewport(0, 0, m_graph->window_width(), m_graph->window_height());
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    m_composite_program->use();

    glm::vec2 pixel_size = glm::vec2(1.0f / float(m_graph->window_width() / 2), 1.0f / float(m_graph->window_height() / 2));
    m_composite_program->set_uniform("u_PixelSize", pixel_size);
    m_composite_program->set_uniform("u_Blend", m_blend);

    if (m_composite_program->set_uniform("s_Color", 0))
        m_color_rt->texture->bind(0);

    if (m_composite_program->set_uniform("s_CoC", 1))
        m_coc_rt->texture->bind(1);

    if (m_composite_program->set_uniform("s_CoC4", 2))
        m_coc4_rt->texture->bind(2);

    if (m_composite_program->set_uniform("s_CoCBlur4", 3))
        m_near_coc_blur4_rt->texture->bind(3);

    if (m_composite_program->set_uniform("s_NearDoF4", 4))
        m_near_fill_dof4_rt->texture->bind(4);

    if (m_composite_program->set_uniform("s_FarDoF4", 5))
        m_far_fill_dof4_rt->texture->bind(5);

    int32_t tex_unit = 0;
    render_fullscreen_triangle(renderer, view, m_composite_program.get(), tex_unit, NODE_USAGE_PER_VIEW_UBO);

    glPopDebugGroup();
}

// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble