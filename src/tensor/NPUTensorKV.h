#pragma once

#include "NPUTensorInner.h"

enum class NPUTensorKVType { KEY, VALUE };

class NPUTensorKV : public NPUTensorInner {
   public:
    NPUTensorKV() = default;
    NPUTensorKV(std::vector<uint32_t> dims, NPUTensorKVType kv_type);
    virtual addr_type get_addr(std::vector<uint32_t> indexes);
    virtual std::vector<addr_type> get_all_addrs();
    uint32_t get_allocated_seq_len();
    void add_token();  // iteration 돌면서 한 토큰씩 추가될 때마다 호출하면 알아서 버퍼 할당.

    NPUTensorKVType _kv_type;
    std::vector<addr_type> _bases;  // KVCache에서 할당받은 row index를 저장
    uint32_t _kv_cache_entry_size;  // 32
    uint32_t _seq_len;              // 현재 KV cache의 seq_len
};