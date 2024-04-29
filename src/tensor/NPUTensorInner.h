#pragma once

#include "../Common.h"

// TensorBufType은 tensor가 어느 buffer에 저장 되어있는지를 의미.
// weight이 ACT에 저장되어 있을 수도 있다.
enum class NPUTensorBufType { WGT, ACT, KV };  // weight, activation, key/value

class NPUTensorInner {
   public:
    NPUTensorInner() = default;
    NPUTensorInner(std::vector<uint32_t> dims, NPUTensorBufType buf_type)
        : _dims(dims), _buf_type(buf_type), _precision(Config::global_config.precision) {}
    virtual addr_type get_addr(std::vector<uint32_t> indexes) = 0;
    virtual std::vector<addr_type> get_all_addrs() = 0;

    addr_type _base_addr;
    std::vector<uint32_t> _dims;
    uint64_t _size;
    NPUTensorBufType _buf_type;
    uint32_t _precision;
};