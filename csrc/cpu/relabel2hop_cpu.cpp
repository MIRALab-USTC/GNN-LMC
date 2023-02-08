#include <torch/script.h>
#include <torch/custom_class.h>
#include <torch/extension.h>

#include <string>
#include <vector>

using namespace torch::indexing;


template <class T>
struct Relabel2hop : torch::CustomClassHolder {
    std::unordered_map<int64_t, int64_t> n_id_map = {};
    torch::Tensor out_rowptr_1hop;
    torch::Tensor out_col_1hop;
    torch::Tensor out_idx_1hop;

    torch::Tensor rowptr_all;
    torch::Tensor col_all;
    torch::optional<torch::Tensor> optional_value_all;

    Relabel2hop(torch::Tensor rowptr, torch::Tensor col,
                        torch::optional<torch::Tensor> optional_value){
        AT_ASSERTM(!rowptr.is_cuda(), "Rowptr tensor must be a CPU tensor");
        AT_ASSERTM(!col.is_cuda(), "Col tensor must be a CPU tensor");
        if (optional_value.has_value()) {
            auto value = optional_value.value();
            AT_ASSERTM(!value.is_cuda(), "Value tensor must be a CPU tensor");
            AT_ASSERTM(value.dim() == 1, "Value tensor must be one-dimensional");
        }
        rowptr_all = rowptr;
        col_all = col;
        optional_value_all = optional_value;
    };

    std::tuple<torch::optional<torch::Tensor>, torch::Tensor>
    relabel_one_hop(torch::Tensor idx, bool bipartite) {

    AT_ASSERTM(!idx.is_cuda(), "Index tensor must be a CPU tensor");
    n_id_map.clear();

    auto rowptr_data = rowptr_all.data_ptr<int64_t>();
    auto col_data = col_all.data_ptr<int64_t>();
    auto idx_data = idx.data_ptr<int64_t>();

    std::vector<int64_t> n_ids;
    std::unordered_map<int64_t, int64_t>::iterator it;

    out_rowptr_1hop = torch::empty(idx.numel() + 1, rowptr_all.options());
    auto out_rowptr_data = out_rowptr_1hop.data_ptr<int64_t>();

    out_rowptr_data[0] = 0;
    int64_t v, w, c, row_start, row_end, offset = 0;
    for (int64_t i = 0; i < idx.numel(); i++) {
        v = idx_data[i];
        n_id_map.insert({v, i}); // n_id_map[v] = i;
        // n_id_map[v] = i;
        offset += rowptr_data[v + 1] - rowptr_data[v];
        out_rowptr_data[i + 1] = offset;
    }

    int64_t numnodes_inbatch = n_id_map.size();
    // int64_t batch_size = idx.numel()

    out_col_1hop = torch::empty(offset, col_all.options());
    auto out_col_data = out_col_1hop.data_ptr<int64_t>();

    torch::optional<torch::Tensor> out_value = torch::nullopt;
    if (optional_value_all.has_value()) {
        out_value = torch::empty(offset, optional_value_all.value().options());

        AT_DISPATCH_ALL_TYPES(optional_value_all.value().scalar_type(), "relabel", [&] {
        auto value_data = optional_value_all.value().data_ptr<scalar_t>();
        auto out_value_data = out_value.value().data_ptr<scalar_t>();

        offset = 0;
        for (int64_t i = 0; i < idx.numel(); i++) {
            v = idx_data[i];
            row_start = rowptr_data[v], row_end = rowptr_data[v + 1];

            for (int64_t j = row_start; j < row_end; j++) {
            w = col_data[j];
            it = n_id_map.find(w);
            if (it == n_id_map.end()) {
                c = numnodes_inbatch + n_ids.size();
                n_id_map[w] = c;
                n_ids.push_back(w);
                out_col_data[offset] = c;
            } else {
                out_col_data[offset] = it->second;
            }
            // if (out_col_data[offset] >= batch_size) {
            //     col_index.push_back(offset);
            // }
            out_value_data[offset] = value_data[j];
            offset++;
            }
        }
        });

    } else {
        offset = 0;
        for (int64_t i = 0; i < idx.numel(); i++) {
        v = idx_data[i];
        row_start = rowptr_data[v], row_end = rowptr_data[v + 1];

        for (int64_t j = row_start; j < row_end; j++) {
            w = col_data[j];
            it = n_id_map.find(w);
            if (it == n_id_map.end()) {
            c = numnodes_inbatch + n_ids.size();
            n_id_map[w] = c;
            n_ids.push_back(w);
            out_col_data[offset] = c;
            } else {
            out_col_data[offset] = it->second;
            }
            offset++;
        }
        }
    }

    if (!bipartite)
        out_rowptr_1hop = torch::cat(
            {out_rowptr_1hop, torch::full({(int64_t)n_ids.size()}, out_col_1hop.numel(),
                                    rowptr_all.options())});

    out_idx_1hop = torch::cat({idx, torch::from_blob(n_ids.data(), {(int64_t)n_ids.size()},
                                            idx.options())});

    return std::make_tuple(out_value, out_idx_1hop);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::optional<torch::Tensor>, torch::Tensor>
    relabel_one_hop_compensation(torch::Tensor idx, bool bipartite) {

    AT_ASSERTM(!idx.is_cuda(), "Index tensor must be a CPU tensor");

    auto rowptr_data = rowptr_all.data_ptr<int64_t>();
    auto col_data = col_all.data_ptr<int64_t>();
    auto idx_data = idx.data_ptr<int64_t>();

    auto out_rowptr_1hop_data = out_rowptr_1hop.data_ptr<int64_t>();
    auto init_offset = out_rowptr_1hop_data[out_rowptr_1hop.numel()-1]; // batch_size = out_rowptr_1hop.numel()

    std::vector<int64_t> n_ids;
    std::unordered_map<int64_t, int64_t>::iterator it;

    auto out_rowptr = torch::empty(idx.numel() + 1, rowptr_all.options());
    auto out_rowptr_data = out_rowptr.data_ptr<int64_t>();

    out_rowptr_data[0] = init_offset;

    // std::unordered_map<int64_t, int64_t> n_id_map_compensate = {};
    int64_t v, w, c, row_start, row_end, offset = init_offset;
    for (int64_t i = 0; i < idx.numel(); i++) {
        v = idx_data[i]; // node id, index
        // n_id_map_compensate.insert({n_id_map[v], i+out_rowptr_1hop.numel()-1}); // n_id_map[v] = i;
        // n_id_map[v] = i;
        offset += rowptr_data[v + 1] - rowptr_data[v];
        out_rowptr_data[i + 1] = offset;
    }

    int64_t numnodes_inbatch = n_id_map.size();

    auto out_col = torch::empty(offset - init_offset, col_all.options());
    auto out_col_data = out_col.data_ptr<int64_t>();

    torch::optional<torch::Tensor> out_value = torch::nullopt;
    if (optional_value_all.has_value()) {
        out_value = torch::empty(offset - init_offset, optional_value_all.value().options());

        AT_DISPATCH_ALL_TYPES(optional_value_all.value().scalar_type(), "relabel", [&] {
        auto value_data = optional_value_all.value().data_ptr<scalar_t>();
        auto out_value_data = out_value.value().data_ptr<scalar_t>();

        offset = 0;
        for (int64_t i = 0; i < idx.numel(); i++) {
            v = idx_data[i];
            row_start = rowptr_data[v], row_end = rowptr_data[v + 1];

            for (int64_t j = row_start; j < row_end; j++) {
            w = col_data[j];
            it = n_id_map.find(w);
            if (it == n_id_map.end()) {
                c = numnodes_inbatch + n_ids.size();
                n_id_map[w] = c;
                n_ids.push_back(w);
                out_col_data[offset] = c;
            } else {
                out_col_data[offset] = it->second;
            }
            out_value_data[offset] = value_data[j];
            offset++;
            }
        };
        // out_value = torch::cat({out_value_1hop, out_value});
        // torch::cat({out_value});
        });
    } else {
        offset = 0;
        for (int64_t i = 0; i < idx.numel(); i++) {
        v = idx_data[i];
        row_start = rowptr_data[v], row_end = rowptr_data[v + 1];

        for (int64_t j = row_start; j < row_end; j++) {
            w = col_data[j];
            it = n_id_map.find(w);
            if (it == n_id_map.end()) {
            c = numnodes_inbatch + n_ids.size();
            n_id_map[w] = c;
            n_ids.push_back(w);
            out_col_data[offset] = c;
            } else {
            out_col_data[offset] = it->second;
            }
            offset++;
        }
        }
    }

    if (!bipartite)
        out_rowptr = torch::cat(
            {out_rowptr_1hop, out_rowptr.index({Slice(1, None, None)}), torch::full({(int64_t)n_ids.size()}, out_col.numel(),
                                    rowptr_all.options())});
    else{
        out_rowptr = torch::cat(
            {out_rowptr_1hop, out_rowptr.index({Slice(1, None, None)}),});
    }
    out_col = torch::cat({out_col_1hop, out_col});
    // idx = torch::from_blob(n_ids.data(), {(int64_t)n_ids.size()},
    //                                         idx.options());
    idx = torch::cat({torch::from_blob(n_ids.data(), {(int64_t)n_ids.size()},
                                            idx.options())});
    // idx = torch::from_blob(n_ids.data(), {(int64_t)n_ids.size()},
    //                                         idx.options());


    return std::make_tuple(out_rowptr, out_col, out_value, idx);
    }



    std::tuple<torch::Tensor, torch::Tensor, torch::optional<torch::Tensor>>
    relabel_one_hop_compensation_rm2hop(torch::Tensor idx, bool bipartite) {

    AT_ASSERTM(!idx.is_cuda(), "Index tensor must be a CPU tensor");

    auto rowptr_data = rowptr_all.data_ptr<int64_t>();
    auto col_data = col_all.data_ptr<int64_t>();
    auto idx_data = idx.data_ptr<int64_t>();

    auto out_rowptr_1hop_data = out_rowptr_1hop.data_ptr<int64_t>();
    auto init_offset = out_rowptr_1hop_data[out_rowptr_1hop.numel()-1]; // batch_size = out_rowptr_1hop.numel()

    std::vector<int64_t> n_ids;
    std::unordered_map<int64_t, int64_t>::iterator it;

    auto out_rowptr = torch::empty(idx.numel() + 1, rowptr_all.options());
    auto out_rowptr_data = out_rowptr.data_ptr<int64_t>();

    out_rowptr_data[0] = init_offset;
    int64_t v, w, c, row_start, row_end, offset_row = init_offset, offset;
    for (int64_t i = 0; i < idx.numel(); i++) {
        v = idx_data[i];
        // n_id_map.insert({v, i}); // n_id_map[v] = i;
        // n_id_map[v] = i;
        offset_row += rowptr_data[v + 1] - rowptr_data[v];
        // out_rowptr_data[i + 1] = offset_row;
    }

    int64_t numnodes_inbatch = n_id_map.size();

    auto out_col = torch::empty(offset_row - init_offset, col_all.options());
    auto out_col_data = out_col.data_ptr<int64_t>();

    torch::optional<torch::Tensor> out_value = torch::nullopt;
    if (optional_value_all.has_value()) {
        out_value = torch::empty(offset_row - init_offset, optional_value_all.value().options());

        AT_DISPATCH_ALL_TYPES(optional_value_all.value().scalar_type(), "relabel", [&] {
        auto value_data = optional_value_all.value().data_ptr<scalar_t>();
        auto out_value_data = out_value.value().data_ptr<scalar_t>();

        offset = 0;
        offset_row = init_offset;
        for (int64_t i = 0; i < idx.numel(); i++) {
            v = idx_data[i];
            row_start = rowptr_data[v], row_end = rowptr_data[v + 1];
            offset_row += row_end - row_start;

            for (int64_t j = row_start; j < row_end; j++) {
                w = col_data[j];
                it = n_id_map.find(w);
                if (it == n_id_map.end()) {
                    offset_row--;
                } else {
                    out_col_data[offset] = it->second;
                    out_value_data[offset] = value_data[j];
                    offset++;
                }
            }
            out_rowptr_data[i + 1] = offset_row;
        }
        });

    } else {
        offset = 0;
        for (int64_t i = 0; i < idx.numel(); i++) {
            v = idx_data[i];
            row_start = rowptr_data[v], row_end = rowptr_data[v + 1];
            offset_row += row_end - row_start;

            for (int64_t j = row_start; j < row_end; j++) {
                w = col_data[j];
                it = n_id_map.find(w);
                if (it == n_id_map.end()) {
                    offset_row--;
                } else {
                    out_col_data[offset] = it->second;
                    offset++;
                }
                out_rowptr_data[i + 1] = offset_row;
            }
        }
    }

    auto col_size = out_rowptr_data[idx.numel()] - out_rowptr_data[0];
    out_col = out_col.index({Slice(0, col_size, None)});

    out_rowptr = torch::cat({out_rowptr_1hop, out_rowptr.index({Slice(1, None, None)}),});
    out_col = torch::cat({out_col_1hop, out_col});

    return std::make_tuple(out_rowptr, out_col, out_value);
    }
};

