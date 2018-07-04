#include "shader_permutator.h"
#include "global_graphics_resources.h"

// -----------------------------------------------------------------------------------------------------------------------------------

ShaderPermutator::ShaderPermutator(std::string vs_src, std::string fs_src) : m_vs_src(vs_src), m_fs_src(fs_src)
{
    
}

// -----------------------------------------------------------------------------------------------------------------------------------

ShaderPermutator::~ShaderPermutator()
{
    
}

// -----------------------------------------------------------------------------------------------------------------------------------

void ShaderPermutator::add_option(const uint32_t& id, const uint32_t& num_possible_values, std::string* possible_values)
{
    
}

// -----------------------------------------------------------------------------------------------------------------------------------

void ShaderPermutator::precompile(const uint64_t& key)
{
    compile(key);
}

// -----------------------------------------------------------------------------------------------------------------------------------

int ShaderPermutator::find_bits_required(int combinations)
{
    int n = 0;
    
    while(true)
    {
        int temp = 0;
        
        for (int i = 0; i < n; i++)
            temp += pow(2, i);
        
        if (temp >= combinations)
            break;
        
        n++;
    }
    
    return n;
}

// -----------------------------------------------------------------------------------------------------------------------------------

uint64_t ShaderPermutator::build_key(const uint32_t& num_values, OptionValue* option_values)
{
    uint64_t key = 0;
    uint32_t pos = 0;
  
    for (int i = 0; i < num_values; i++)
    {
        Option& opt = m_options[option_values[i].first];
        int num_bits = find_bits_required(opt.num_possible_values);
        
        uint64_t temp = option_values[i].second;
        key |= temp << pos;
    
        pos += num_bits;
    }
    
    return key;
}

// -----------------------------------------------------------------------------------------------------------------------------------

dw::Program* ShaderPermutator::lookup(const uint64_t& key)
{
    if (m_program_cache.find(key) == m_program_cache.end())
        return compile(key);
    else
        return m_program_cache[key];
}

// -----------------------------------------------------------------------------------------------------------------------------------

dw::Program* ShaderPermutator::compile(const uint64_t& key)
{
    uint32_t pos = 0;
    std::vector<std::string> defines;
    
    for (int i = 0; i < m_options.size(); i++)
    {
        int num_bits = find_bits_required(m_options[i].num_possible_values);
        int value = (key >> pos) & (int(pow(2.0f, float(num_bits))) - 1);
        
        std::string define = m_options[i].possible_values[value];
        
        if (define != EMPTY_OPTION)
            defines.push_back(define);
    }
    
    dw::Shader* vs = compile_shader(GL_VERTEX_SHADER, m_vs_src, defines);
    dw::Shader* fs = compile_shader(GL_FRAGMENT_SHADER, m_fs_src, defines);
    
    dw::Shader* shaders[] = { vs, fs };
    
    dw::Program* program = new dw::Program(2, shaders);
    m_program_cache[key] = program;
    
    return program;
}

// -----------------------------------------------------------------------------------------------------------------------------------

dw::Shader* ShaderPermutator::compile_shader(GLenum type, std::string src, std::vector<std::string>& defines)
{
    for (auto& define : defines)
    {
        src += "#define ";
        src += define;
        src += "\n";
    }
    
    src += "\n";
    
    return new dw::Shader(type, src);
}

// -----------------------------------------------------------------------------------------------------------------------------------
