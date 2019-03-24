#pragma once

#include <string>
#include <vector>
#include <stdint.h>

namespace nimble
{
struct BoolParameter
{
    bool*       ptr;
    std::string name;
};

struct IntParameter
{
    int32_t*    ptr;
    std::string name;
    int32_t     min = 0;
    int32_t     max = 0;
};

struct FloatParameter
{
    float*      ptr;
    std::string name;
    float       min = 0.0f;
    float       max = 0.0f;
};

class Parameterizable
{
public:
    Parameterizable();
    ~Parameterizable();
    void            set_bool_parameter(const std::string& name, bool value);
    void            set_int_parameter(const std::string& name, int32_t value);
    void            set_float_parameter(const std::string& name, float value);
    BoolParameter*  bool_parameters(int32_t& count);
    IntParameter*   int_parameters(int32_t& count);
    FloatParameter* float_parameters(int32_t& count);

protected:
    void register_bool_parameter(const std::string& name, bool& parameter);
    void register_int_parameter(const std::string& name, int32_t& parameter, int32_t min = 0, int32_t max = 0);
    void register_float_parameter(const std::string& name, float& parameter, float min = 0.0f, float max = 0.0f);

protected:
    std::vector<BoolParameter>  m_bool_parameters;
    std::vector<IntParameter>   m_int_parameters;
    std::vector<FloatParameter> m_float_parameters;
};
} // namespace nimble