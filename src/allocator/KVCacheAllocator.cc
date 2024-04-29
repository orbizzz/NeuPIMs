#include "AddressAllocator.h"

KVCacheAlloc::KVCacheAlloc()
    : _kv_cache_size(0), _kv_cache_limit(0), _kv_cache_entry_size(0), _base_addr(0), _base_row(0) {}

void KVCacheAlloc::init(addr_type base_addr) {
    _mode = Config::global_config.run_mode;
    if (_mode == RunMode::NPU_ONLY) {  // NPU only mode
        init_npu_layout(base_addr);
    } else if (_mode == RunMode::NPU_PIM) {
        init_pim_layout(base_addr);
    } else {
        ast(0);
    }
}

/**
 * Initialize NPU memory layout.
 * Firt, allocate the whole memory considering predetermined cache size.
 * Cache entry is consisted of 32 key and value, both sizes (d_k).
 *  Note that cache is saved as d_k granularity, not E
 * the memory layout of saved cache is (h,l,d_k).
 * this is because it is because consecutive memory access is faster,
 * so adjacent latent vector at certain head should be loaded faster.
 */
void KVCacheAlloc::init_npu_layout(addr_type base_addr) {
    uint32_t max_active_reqs = Config::global_config.max_active_reqs;
    uint32_t max_seq_len = Config::global_config.max_seq_len;
    uint32_t h = Config::global_config.model_n_head;
    uint32_t d_k = Config::global_config.model_n_embd / h;
    uint32_t precision = Config::global_config.precision;

    _base_addr = base_addr;
    _kv_cache_entry_size = 32;  // seq_len 32개 당 하나씩 할당하자.
    _kv_cache_size = max_active_reqs * max_seq_len * h * d_k * precision;
    ast(_base_addr + _kv_cache_size < Config::global_config.HBM_size);

    addr_type next_addr = _base_addr;
    // 저장 가능한 seq_len 수 / block 1개 당 seq_len (block은 32 * d_k elements)
    // = HBM의 KV cache block의 수
    uint64_t num_kv_cache_entries = max_active_reqs * max_seq_len * h / _kv_cache_entry_size;

    // 현재 NPUTensor가 최대 2차원까지의 inner_tensor를 지원하기 때문에
    // [h, l, d_k]를 h개의 [l, d_k]로 나누어 할당한다. ??? 왜 h개의 dk를 l개? (l,h,dk)
    // is using h,l,dk is acceptible in memory perspective?
    //  연속적인 할당이 read bandwidth를 더 활용 잘할듯. okay
    for (int i = 0; i < num_kv_cache_entries; ++i) {
        _kv_cache.push_back(next_addr);
        next_addr += _kv_cache_entry_size * d_k * precision;  // 32 seq_len * d_k * precision
    }
}

void KVCacheAlloc::init_pim_layout(addr_type base_addr) {
    // =rows of matrix in a DRAM PIM row
    constexpr uint32_t row_per_bank = 32768;
    constexpr uint32_t row_offset = 21;
    constexpr uint64_t mask = ~((1 << row_offset) - 1);  // 0x1111(64-21개)0000(21개)
    _dram_col_size = 1024;
    _num_ele_per_row = _dram_col_size / Config::global_config.precision;  // 512
    _bank_per_ch = Config::global_config.dram_banks_per_ch;
    _dram_channels = Config::global_config.dram_channels;

    base_addr = base_addr & mask;  // 사용되고 있는 가장 마지막 row index를 추출
    base_addr = base_addr + (1 << row_offset);  // 다음 row index로 이동

    _base_addr = base_addr;
    _base_row = base_addr >> row_offset;  // row index만 추출

    // _rows: channel -> row idx
    uint32_t free_rows_size = row_per_bank - _base_row;
    for (int i = 0; i < _dram_channels; ++i) {
        _rows.push_back(std::make_shared<std::deque<uint64_t>>());
        for (int j = 0; j < free_rows_size; ++j) {
            if (_base_row + j < row_per_bank) _rows[i]->push_back(_base_row + j);
        }
    }
}

// [bank per ch, d_k]만큼의 공간을 할당해 return 한다.
// 이걸 h번 반복하면 [h, bank per ch, d_k]만큼의 공간을 할당한 것이 된다.
addr_type KVCacheAlloc::allocate() {
    ast(_mode == RunMode::NPU_ONLY);
    ast(_kv_cache.size() > 0);
    addr_type addr = _kv_cache.front();
    _kv_cache.pop_front();
    return addr;
}

addr_type KVCacheAlloc::allocate(uint64_t ch) {
    ast(_mode == RunMode::NPU_PIM);
    ast(_rows[ch]->size() > 0);
    addr_type row = _rows[ch]->front();
    _rows[ch]->pop_front();
    return row;  // free row를 return.
}

void KVCacheAlloc::free(addr_type addr) {
    ast(_mode == RunMode::NPU_ONLY);
    _kv_cache.push_back(addr);
}

void KVCacheAlloc::free(uint32_t ch, uint64_t row) {
    ast(_mode == RunMode::NPU_PIM);
    _rows[ch]->push_back(row);
}