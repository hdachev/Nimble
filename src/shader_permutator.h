#pragma once

#include <unordered_map>
#include <string>
#include <vector>
#include "ogl.h"

namespace nimble
{
    #define EMPTY_OPTION "EMPTY_OPTION"

    // Index into m_options using the option ID.
    using OptionValue = std::pair<uint32_t, uint32_t>;

    class ShaderPermutator
    {
    public:
        ShaderPermutator(std::string vs_src, std::string fs_src);
        ~ShaderPermutator();
        void add_option(const uint32_t& id, const uint32_t& num_possible_values, std::string* possible_values);
        void precompile(const uint64_t& key);
        uint64_t build_key(const uint32_t& num_values, OptionValue* option_values);
        Program* lookup(const uint64_t& key);
        
    private:
        int find_bits_required(int combinations);
        Program* compile(const uint64_t& key);
        Shader* compile_shader(GLenum type, std::string src, std::vector<std::string>& defines);
        
    private:
        struct Option
        {
            std::string name;
            uint32_t num_possible_values;
            std::vector<std::string> possible_values;
        };
        
        std::string m_vs_src;
        std::string m_fs_src;
        std::vector<Option> m_options;
        std::unordered_map<uint64_t, Program*> m_program_cache;
    };
}
