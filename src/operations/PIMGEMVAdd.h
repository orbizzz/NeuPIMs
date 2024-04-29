
/*
>>> GEMV Add tiling (LsxV) <<<

page_size = 512; // dram row 하나에 들어가는 parameter 수
banks_per_channel = 32;
datas_per_comp_cmd = 16;


// chunks_per_head = ceil(seq_len / page_size);
tiles_per_chunk = ceil(dk / banks_per_channel);

for (auto req: batch) {
    chunks = ceil(seq_len / page_size);
    for (auto head: heads) {
        for (auto chunk: chunks) {
            >>> gsheo: P_HEADER
            >>> gsheo: GWRITE
            num_comps = chunk.is_last_chunk_in_head ? (seq_len % page_size)/16 : 32;
            for (auto tile: tiles) {
                >>> gsheo: P_HEADER (num_comps, num_readres = 1)
                >>> gsheo: COMP * num_comps
                // -- activation --
                >>> gsheo: READRES * 1 (sram_readres_result_addr_ head# chunk# tile#)
            }
        }
        if (chunks > 1) {
            같은 head 내의 chunk들의 같은 tile에서 나온 결과들을 더해줌.
            for (auto tile: tiles) {
                for (auto chunk: chunks) {
                    src_addrs.push(head, chunk, tile)
                }
                // -- compute --
                >>> gsheo: Add
                    src_addrs:
                        readres of (head#1, chunk#1, tile#1)
                        readres of (head#1, chunk#2, tile#1)
                        ..
                        readres of (head#1, chunk#chunks_per_heads, tile#1)
            }
        }
        --- save outputs ---
        >>> gsheo: WRITE (dk * 1) activation params
    }
}

*/

#pragma once
#include "../tensor/NPUTensor.h"
#include "../tensor/PIMTensor.h"
#include "Operation.h"

class PIMGEMVAdd : public Operation {
   public:
    PIMGEMVAdd(std::string name);

    std::vector<Ptr<BTensor>> get_outputs(std::vector<Ptr<BTensor>> inputs) override;

    uint32_t _batch_size;
    std::vector<Ptr<NPUTensor>> _logits;
    std::vector<Ptr<PIMTensor>> _vs;

    std::vector<uint32_t> _inner_loop;
    std::vector<uint32_t> _outer_loop;

    // model spec
    uint32_t _nh;
    uint32_t _dk;

    // memory spec
    uint32_t _page_size;
    uint32_t _banks_per_channel;

    uint32_t _tiles_per_chunk;
    uint32_t _datas_per_comp_cmd;

    void calculate_loops();
    void initialize_tiles();
    Tile initialize_instructions();
    uint32_t sram_size_needed();
};