#pragma once

#include "BTensor.h"

enum class PIMTensorKVType { KEY, VALUE };

class PIMTensor : public BTensor {
   public:
    PIMTensor() = default;
    PIMTensor(std::string name, uint32_t ch, std::vector<uint32_t> dims, PIMTensorKVType kv_type,
              bool produced);
    ~PIMTensor() = default;

    virtual addr_type get_addr(std::vector<uint32_t> indexes) override;
    virtual std::vector<addr_type> get_all_addrs() override;
    virtual void add_token()
        override;  // iteration 돌면서 한 토큰씩 추가될 때마다 호출하면 알아서 버퍼 할당.

    uint32_t get_allocated_seq_len();
    uint32_t get_num_rows();
    uint32_t get_channel();
    std::vector<uint64_t> get_rows();

    PIMTensorKVType _kv_type;
    uint32_t _bank_per_ch;
    uint32_t _E;  // gsheo: 필요한가?
    uint32_t _num_ele_per_row;
    //  여기서 row는 DRAM row를 의미한다.
    // seq_len가 늘어나서 추가로 alloc 할 때 한 번에 몇 개의 row씩 alloc 해야 하는지.
    uint32_t _num_rows_per_alloc;

    uint32_t _ch;                 // DRAM channel
    std::vector<uint64_t> _rows;  // KVCache에서 할당받은 row index를 저장
    uint32_t _seq_len;
};