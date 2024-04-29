#pragma once
#include "Common.h"

// TensorBufType은 tensor가 어느 buffer에 저장 되어있는지를 의미.
// weight이 ACT에 저장되어 있을 수도 있다.
enum class TensorBufType { WGT, ACT, KV };  // weight, activation, key/value

class Tensor2D {
   public:
    Tensor2D() = default;
    Tensor2D(std::vector<uint32_t> dims, TensorBufType buf_type);
    addr_type get_addr(std::vector<uint32_t> indexes);
    std::vector<std::shared_ptr<Tensor2D>> split_by_row(std::vector<uint32_t> row_dims);

    addr_type _base_addr;
    std::vector<uint32_t> _dims;
    uint64_t _size;
    TensorBufType _buf_type;
};