// Direct loading of vanilla HuggingFace safetensors models.
//
// See llama-safetensors.h for the design. Only the qwen35 architecture is
// supported (the tensor name map below mirrors conversion/qwen.py).

#include "llama-safetensors.h"

#include "llama-impl.h"
#include "llama-mmap.h"
#include "llama.h"

#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"
#include "gguf.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <sstream>

namespace fs = std::filesystem;
using json = nlohmann::json;

static const size_t MIB = 1024 * 1024;

// ---------------------------------------------------------------------------
// helpers

static std::string read_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error(format("cannot open file: %s", path.c_str()));
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static json read_json(const std::string & path) {
    try {
        return json::parse(read_file(path));
    } catch (const std::exception & err) {
        throw std::runtime_error(format("failed to parse %s: %s", path.c_str(), err.what()));
    }
}

// ---------------------------------------------------------------------------
// path detection

bool llama_safetensors_loader::is_safetensors_path(const char * path) {
    const std::string p = path;
    std::error_code ec;
    if (fs::is_directory(p, ec)) {
        if (!fs::exists(p + "/config.json", ec)) {
            return false;
        }
        for (const auto & entry : fs::directory_iterator(p, ec)) {
            if (entry.path().extension() == ".safetensors") {
                return true;
            }
        }
        return false;
    }
    return p.size() > 12 && p.compare(p.size() - 12, 12, ".safetensors") == 0;
}

// probe every registered device with a dummy fp8 mul_mat
bool llama_safetensors_loader::has_fp8_device() {
    ggml_init_params ip = { ggml_tensor_overhead() * 8, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F8_E4M3, 256, 16);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 1);
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);

    bool ok = false;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        if (ggml_backend_dev_supports_op(ggml_backend_dev_get(i), y)) {
            ok = true;
            break;
        }
    }
    ggml_free(ctx);
    return ok;
}

bool llama_safetensors_loader::has_fp8_tensors() const {
    for (const auto & [gname, m] : gguf_map) {
        (void) gname;
        if (m.type == GGML_TYPE_F8_E4M3) {
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// construction

llama_safetensors_loader::llama_safetensors_loader(const char * path) {
    std::string p = path;
    if (fs::is_directory(p)) {
        dir_path = p;
        if (dir_path.back() != '/') {
            dir_path += '/';
        }
    } else {
        // single .safetensors file: use its parent dir as the model dir
        dir_path = fs::path(p).parent_path().string();
        if (!dir_path.empty() && dir_path.back() != '/') {
            dir_path += '/';
        }
    }
    vocab_path = dir_path + "tokenizer.gguf";

    parse_config();
    parse_files();
    build_permutations();
    build_mapping();
}

llama_safetensors_loader::~llama_safetensors_loader() = default;

void llama_safetensors_loader::parse_config() {
    const json cfg = read_json(dir_path + "config.json");

    const auto & archs = cfg.value("architectures", json::array());
    const bool ok_arch = !archs.empty() &&
        (archs[0] == "Qwen3_5ForConditionalGeneration" || archs[0] == "Qwen3_5ForCausalLM");
    if (!ok_arch) {
        throw std::runtime_error(format("unsupported model architecture '%s' - the direct safetensors loader supports qwen35 only",
                archs.empty() ? "?" : archs[0].get<std::string>().c_str()));
    }

    const json tc = cfg.value("text_config", cfg);

    n_embd        = tc.at("hidden_size");
    n_layer       = tc.at("num_hidden_layers");
    n_ff          = tc.at("intermediate_size");
    n_head        = tc.at("num_attention_heads");
    n_head_kv     = tc.at("num_key_value_heads");
    n_ctx         = tc.at("max_position_embeddings");
    n_vocab       = tc.at("vocab_size");
    rms_eps       = tc.at("rms_norm_eps");

    ssm_d_conv    = tc.at("linear_conv_kernel_dim");
    ssm_d_state   = tc.at("linear_key_head_dim");
    ssm_n_group   = tc.at("linear_num_key_heads");
    ssm_dt_rank   = tc.at("linear_num_value_heads");
    ssm_d_inner   = tc.at("linear_value_head_dim").get<int64_t>() * ssm_dt_rank;

    n_layer_nextn = tc.value("mtp_num_hidden_layers", 0);

    const int64_t head_dim = tc.value("head_dim", 0) != 0 ? tc.value("head_dim", 0) : n_embd / n_head;
    n_embd_head_k = head_dim;

    const json rp = tc.value("rope_parameters", json::object());
    rope_freq_base = rp.value("rope_theta", 1.0e7f);
    const float partial_rotary_factor = rp.value("partial_rotary_factor", 0.25f);
    rope_dim = (int64_t) (head_dim * partial_rotary_factor);

    // Qwen3.5 always uses interleaved MRoPE with the default [11, 11, 10] section
    std::array<int32_t, 4> section = { 11, 11, 10, 0 };
    if (rp.contains("mrope_section")) {
        const auto & ms = rp["mrope_section"];
        for (size_t i = 0; i < ms.size() && i < 3; ++i) {
            section[i] = ms[i];
        }
    }
    memcpy(mrope_section, section.data(), sizeof(mrope_section));
}

void llama_safetensors_loader::parse_files() {
    // list the safetensors files: from the index, or a single file
    std::vector<std::string> names;
    const std::string index_path = dir_path + "model.safetensors.index.json";
    if (fs::exists(index_path)) {
        const json idx = read_json(index_path);
        if (!idx.contains("weight_map")) {
            throw std::runtime_error(format("invalid safetensors index: %s", index_path.c_str()));
        }
        for (const auto & [name, file] : idx["weight_map"].items()) {
            (void) name;
            const std::string fname = file.get<std::string>();
            if (std::find(names.begin(), names.end(), fname) == names.end()) {
                names.push_back(fname);
            }
        }
    } else {
        for (const auto & entry : fs::directory_iterator(dir_path)) {
            if (entry.path().extension() == ".safetensors") {
                names.push_back(entry.path().filename().string());
            }
        }
        if (names.empty()) {
            throw std::runtime_error(format("no .safetensors files found in %s", dir_path.c_str()));
        }
    }
    std::sort(names.begin(), names.end());

    for (const auto & name : names) {
        const std::string path = dir_path + name;

        auto file = std::make_unique<llama_file>(path.c_str(), "rb");
        uint8_t len_buf[8];
        file->read_raw(len_buf, 8);
        uint64_t header_len;
        memcpy(&header_len, len_buf, 8);

        std::string header_str(header_len, '\0');
        file->read_raw(&header_str[0], header_len);

        json header;
        try {
            header = json::parse(header_str);
        } catch (const std::exception & err) {
            throw std::runtime_error(format("corrupt safetensors header in %s: %s", path.c_str(), err.what()));
        }
        const size_t data_off = 8 + header_len;

        auto f = std::make_unique<st_file>();
        f->path = path;

        ll_mmaps.emplace_back(std::make_unique<llama_mmap>(file.get(), -1, false));
        f->addr = (const uint8_t *) ll_mmaps.back()->addr();
        ll_files.emplace_back(std::move(file));
        files[name] = std::move(f);

        for (const auto & [tname, tinfo] : header.items()) {
            if (tname == "__metadata__") {
                continue;
            }
            st_tensor t;
            t.file   = name;
            t.off    = data_off + (size_t) tinfo.at("data_offsets")[0].get<int64_t>();
            t.nbytes = (size_t) (tinfo.at("data_offsets")[1].get<int64_t>() - tinfo.at("data_offsets")[0].get<int64_t>());
            t.dtype  = tinfo.at("dtype");
            t.shape  = tinfo.at("shape").get<std::vector<int64_t>>();
            if (t.dtype != "F8_E4M3" && t.dtype != "BF16" && t.dtype != "F32") {
                throw std::runtime_error(format("unsupported tensor dtype '%s' for '%s' - supported: F8_E4M3, BF16, F32", t.dtype.c_str(), tname.c_str()));
            }
            if (t.off + t.nbytes > ll_mmaps.back()->size()) {
                throw std::runtime_error(format("tensor '%s' in %s is out of file bounds", tname.c_str(), name.c_str()));
            }
            hf_tensors[tname] = std::move(t);
        }
    }
}

void llama_safetensors_loader::build_permutations() {
    // linear attention has num_k_heads < num_v_heads; the HF weights store the
    // V heads grouped by K head [G0_v0..v{r-1}, G1_v0..], ggml needs them
    // tiled [v0_G0, v0_G1, ..., v1_G0, ...]. See conversion/qwen.py.
    const int64_t nk   = ssm_n_group;              // 16
    const int64_t nvpk = ssm_dt_rank / ssm_n_group; // 2

    // head_dim = 1 (A_log, dt_bias, in_proj_a/b rows)
    v_perm_1.resize(ssm_dt_rank);
    for (int64_t k = 0; k < nk; ++k) {
        for (int64_t vp = 0; vp < nvpk; ++vp) {
            v_perm_1[vp * nk + k] = k * nvpk + vp;
        }
    }

    // head_dim = 128 (qkv/z rows, out_proj columns, conv1d channels)
    const int64_t hd = ssm_d_state;
    v_perm_128.resize(ssm_dt_rank * hd);
    for (int64_t k = 0; k < nk; ++k) {
        for (int64_t vp = 0; vp < nvpk; ++vp) {
            for (int64_t h = 0; h < hd; ++h) {
                v_perm_128[vp * (nk * hd) + k * hd + h] = k * (nvpk * hd) + vp * hd + h;
            }
        }
    }

    // qkv/conv1d rows: q and k rows are unchanged, only the v rows are reordered
    const int64_t qk_rows = 2 * nk * hd;
    v_perm_128_v.resize(qk_rows + ssm_dt_rank * hd);
    for (int64_t r = 0; r < qk_rows; ++r) {
        v_perm_128_v[r] = r;
    }
    for (int64_t r = 0; r < (int64_t) v_perm_128.size(); ++r) {
        v_perm_128_v[qk_rows + r] = qk_rows + v_perm_128[r];
    }
}

// ---------------------------------------------------------------------------
// HF -> GGUF tensor mapping (mirrors conversion/qwen.py)

bool llama_safetensors_loader::map_weight(const std::string & hf, const st_tensor & t, std::string & gname, st_mapping & m) const {
    // V-head reorder applies to linear-attention tensors when K-heads != V-heads;
    // the fp8 path re-blocks and reorders in one pass (fill_fp8), the float paths
    // reorder rows/columns (TF_REORDER)
    auto set_reorder = [&](const std::vector<int64_t> * rows, const std::vector<int64_t> * cols) {
        if (t.dtype == "F8_E4M3") {
            m.transform = TF_FP8;
            m.type = GGML_TYPE_F8_E4M3;
        } else {
            m.transform = TF_REORDER;
            m.type = GGML_TYPE_BF16;
        }
        m.row_perm = rows;
        m.col_perm = cols;
    };
    auto set_plain = [&](int transform = TF_NONE) {
        m.transform = transform;
        m.type = t.dtype == "F8_E4M3" ? GGML_TYPE_F8_E4M3 : GGML_TYPE_BF16;
    };
    auto set_fp8 = [&]() {
        m.transform = TF_FP8;
        m.type = GGML_TYPE_F8_E4M3;
    };

    const std::string pfx = "model.language_model.";
    if (hf.rfind(pfx, 0) == 0) {
        // handled below
    } else if (hf.rfind("mtp.", 0) == 0) {
        // MTP tensors have no model.language_model. prefix
        const int64_t il = n_layer; // MTP block is appended after the trunk
        char buf[128];
        const std::string s = hf.substr(4);
        if (s == "fc.weight") {
            snprintf(buf, sizeof(buf), "blk.%lld.nextn.eh_proj.weight", (long long) il);
            gname = buf; set_plain(); return true;
        }
        if (s == "pre_fc_norm_embedding.weight") {
            snprintf(buf, sizeof(buf), "blk.%lld.nextn.enorm.weight", (long long) il);
            gname = buf; set_plain(TF_NORM_P1); return true;
        }
        if (s == "pre_fc_norm_hidden.weight") {
            snprintf(buf, sizeof(buf), "blk.%lld.nextn.hnorm.weight", (long long) il);
            gname = buf; set_plain(TF_NORM_P1); return true;
        }
        if (s == "norm.weight") {
            snprintf(buf, sizeof(buf), "blk.%lld.nextn.shared_head_norm.weight", (long long) il);
            gname = buf; set_plain(TF_NORM_P1); return true;
        }
        if (s.rfind("layers.", 0) == 0) {
            const size_t p1 = 7;
            const size_t p2 = s.find('.', p1);
            const int64_t i = atoll(s.substr(p1, p2 - p1).c_str());
            const std::string rest = s.substr(p2 + 1);
            if (i != 0) {
                return false;
            }
            if (rest == "input_layernorm.weight") {
                snprintf(buf, sizeof(buf), "blk.%lld.attn_norm.weight", (long long) il);
                gname = buf; set_plain(TF_NORM_P1); return true;
            }
            if (rest == "post_attention_layernorm.weight") {
                snprintf(buf, sizeof(buf), "blk.%lld.post_attention_norm.weight", (long long) il);
                gname = buf; set_plain(TF_NORM_P1); return true;
            }
            if (rest == "mlp.gate_proj.weight") {
                snprintf(buf, sizeof(buf), "blk.%lld.ffn_gate.weight", (long long) il);
                gname = buf; set_plain(); return true;
            }
            if (rest == "mlp.up_proj.weight") {
                snprintf(buf, sizeof(buf), "blk.%lld.ffn_up.weight", (long long) il);
                gname = buf; set_plain(); return true;
            }
            if (rest == "mlp.down_proj.weight") {
                snprintf(buf, sizeof(buf), "blk.%lld.ffn_down.weight", (long long) il);
                gname = buf; set_plain(); return true;
            }
            if (rest.rfind("self_attn.", 0) == 0) {
                const std::string st = rest.substr(10);
                if (st == "q_proj.weight") {
                    snprintf(buf, sizeof(buf), "blk.%lld.attn_q.weight", (long long) il);
                    gname = buf; set_plain(); return true;
                }
                if (st == "k_proj.weight" || st == "v_proj.weight") {
                    snprintf(buf, sizeof(buf), "blk.%lld.attn_%s.weight", (long long) il, st[0] == 'k' ? "k" : "v");
                    gname = buf; set_plain(); return true;
                }
                if (st == "o_proj.weight") {
                    snprintf(buf, sizeof(buf), "blk.%lld.attn_output.weight", (long long) il);
                    gname = buf; set_plain(); return true;
                }
                if (st == "q_norm.weight" || st == "k_norm.weight") {
                    snprintf(buf, sizeof(buf), "blk.%lld.attn_%c_norm.weight", (long long) il, st[0]);
                    gname = buf; set_plain(TF_NORM_P1); return true;
                }
                return false;
            }
        }
        return false;
    } else {
        return false; // vision and any other prefixes are skipped
    }
    const std::string rel = hf.substr(pfx.size());

    if (rel == "embed_tokens.weight") {
        gname = "token_embd.weight";
        set_plain();
        return true;
    }
    if (rel == "norm.weight") {
        gname = "output_norm.weight";
        set_plain(TF_NORM_P1);
        return true;
    }

    if (rel.rfind("layers.", 0) == 0) {
        const size_t p1 = 7;
        const size_t p2 = rel.find('.', p1);
        const int64_t i = atoll(rel.substr(p1, p2 - p1).c_str());
        const std::string rest = rel.substr(p2 + 1);

        char buf[128];
        if (i >= n_layer) {
            return false;
        }
        // shared tensors
        if (rest == "input_layernorm.weight") {
            snprintf(buf, sizeof(buf), "blk.%lld.attn_norm.weight", (long long) i);
            gname = buf; set_plain(TF_NORM_P1); return true;
        }
        if (rest == "post_attention_layernorm.weight") {
            snprintf(buf, sizeof(buf), "blk.%lld.post_attention_norm.weight", (long long) i);
            gname = buf; set_plain(TF_NORM_P1); return true;
        }
        if (rest == "mlp.gate_proj.weight") {
            snprintf(buf, sizeof(buf), "blk.%lld.ffn_gate.weight", (long long) i);
            gname = buf; set_plain(); return true;
        }
        if (rest == "mlp.up_proj.weight") {
            snprintf(buf, sizeof(buf), "blk.%lld.ffn_up.weight", (long long) i);
            gname = buf; set_plain(); return true;
        }
        if (rest == "mlp.down_proj.weight") {
            snprintf(buf, sizeof(buf), "blk.%lld.ffn_down.weight", (long long) i);
            gname = buf; set_plain(); return true;
        }
        if (rest.rfind("self_attn.", 0) == 0) {
            const std::string s = rest.substr(10);
            if (s == "q_proj.weight") {
                snprintf(buf, sizeof(buf), "blk.%lld.attn_q.weight", (long long) i);
                gname = buf; set_plain(); return true;
            }
            if (s == "k_proj.weight" || s == "v_proj.weight") {
                snprintf(buf, sizeof(buf), "blk.%lld.attn_%s.weight", (long long) i, s[0] == 'k' ? "k" : "v");
                gname = buf; set_plain(); return true;
            }
            if (s == "o_proj.weight") {
                snprintf(buf, sizeof(buf), "blk.%lld.attn_output.weight", (long long) i);
                gname = buf; set_plain(); return true;
            }
            if (s == "q_norm.weight" || s == "k_norm.weight") {
                snprintf(buf, sizeof(buf), "blk.%lld.attn_%c_norm.weight", (long long) i, s[0]);
                gname = buf; set_plain(TF_NORM_P1); return true;
            }
            return false;
        }
        if (rest.rfind("linear_attn.", 0) == 0) {
            const std::string s = rest.substr(12);
            if (s == "in_proj_qkv.weight") {
                snprintf(buf, sizeof(buf), "blk.%lld.attn_qkv.weight", (long long) i);
                gname = buf; set_reorder(&v_perm_128_v, nullptr); return true;
            }
            if (s == "in_proj_z.weight") {
                snprintf(buf, sizeof(buf), "blk.%lld.attn_gate.weight", (long long) i);
                gname = buf; set_reorder(&v_perm_128, nullptr); return true;
            }
            if (s == "out_proj.weight") {
                snprintf(buf, sizeof(buf), "blk.%lld.ssm_out.weight", (long long) i);
                gname = buf; set_reorder(nullptr, &v_perm_128); return true;
            }
            if (s == "in_proj_a.weight") {
                snprintf(buf, sizeof(buf), "blk.%lld.ssm_alpha.weight", (long long) i);
                gname = buf; set_reorder(&v_perm_1, nullptr); return true;
            }
            if (s == "in_proj_b.weight") {
                snprintf(buf, sizeof(buf), "blk.%lld.ssm_beta.weight", (long long) i);
                gname = buf; set_reorder(&v_perm_1, nullptr); return true;
            }
            if (s == "conv1d.weight") {
                snprintf(buf, sizeof(buf), "blk.%lld.ssm_conv1d.weight", (long long) i);
                gname = buf; set_reorder(&v_perm_128_v, nullptr); return true;
            }
            if (s == "A_log") {
                snprintf(buf, sizeof(buf), "blk.%lld.ssm_a", (long long) i);
                gname = buf; m.transform = TF_A_LOG; m.row_perm = &v_perm_1; m.type = GGML_TYPE_F32;
                return true;
            }
            if (s == "dt_bias") {
                snprintf(buf, sizeof(buf), "blk.%lld.ssm_dt.bias", (long long) i);
                gname = buf; set_reorder(&v_perm_1, nullptr); return true;
            }
            if (s == "norm.weight") {
                snprintf(buf, sizeof(buf), "blk.%lld.ssm_norm.weight", (long long) i);
                gname = buf; set_plain(); return true;
            }
            return false;
        }
        return false;
    }

    return false;
}

void llama_safetensors_loader::build_mapping() {
    for (const auto & [hf, t] : hf_tensors) {
        // scale_inv tensors are consumed by their fp8 companions
        const std::string suffix = ".weight_scale_inv";
        if (hf.size() > suffix.size() && hf.compare(hf.size() - suffix.size(), suffix.size(), suffix) == 0) {
            continue;
        }
        std::string gname;
        st_mapping m;
        m.hf = hf;
        m.src_f32 = t.dtype == "F32";
        if (!map_weight(hf, t, gname, m)) {
            continue;
        }

        // ggml dims: fastest dim first (HF stores [out, in] row-major)
        if (t.shape.size() == 1) {
            m.ne = { t.shape[0] };
        } else if (t.shape.size() == 2) {
            m.ne = { t.shape[1], t.shape[0] };
        } else if (t.shape.size() == 3) {
            // conv1d [C, 1, K] -> [K, C]
            m.ne = { t.shape[2], t.shape[0] };
        } else {
            throw std::runtime_error(format("unsupported tensor rank %zu for '%s'", t.shape.size(), hf.c_str()));
        }

        // mirror conversion/base.py: 1D tensors, *_norm.weight and conv1d are always F32
        if (m.type == GGML_TYPE_BF16) {
            const bool is_1d     = m.ne.size() == 1;
            const bool is_norm   = gname.size() >= 12 && gname.compare(gname.size() - 12, 12, "_norm.weight") == 0;
            const bool is_conv1d = gname.find("ssm_conv1d") != std::string::npos;
            if (is_1d || is_norm || is_conv1d) {
                m.type = GGML_TYPE_F32;
            }
        }

        if (m.type == GGML_TYPE_F8_E4M3) {
            // the fp8 kernel consumes 128-column blocks; ggml asserts ne[0] % 128
            if (m.ne[0] % 128 != 0) {
                throw std::runtime_error(format("fp8 tensor '%s' has %lld columns, must be a multiple of 128", gname.c_str(), (long long) m.ne[0]));
            }
            // the companion scale is stored as <name>.weight_scale_inv
            m.hf_scale = hf.substr(0, hf.size() - 7) + ".weight_scale_inv";
            if (hf_tensors.find(m.hf_scale) == hf_tensors.end()) {
                throw std::runtime_error(format("missing fp8 scale tensor '%s'", m.hf_scale.c_str()));
            }
        }

        gguf_map[gname] = std::move(m);
    }
}

// ---------------------------------------------------------------------------
// metadata synthesis

static void copy_kv(struct gguf_context * dst, struct gguf_context * src, int64_t kid) {
    const char * key = gguf_get_key(src, kid);
    switch (gguf_get_kv_type(src, kid)) {
        case GGUF_TYPE_UINT8:   gguf_set_val_u8  (dst, key, gguf_get_val_u8  (src, kid)); break;
        case GGUF_TYPE_INT8:    gguf_set_val_i8  (dst, key, gguf_get_val_i8  (src, kid)); break;
        case GGUF_TYPE_UINT16:  gguf_set_val_u16 (dst, key, gguf_get_val_u16 (src, kid)); break;
        case GGUF_TYPE_INT16:   gguf_set_val_i16 (dst, key, gguf_get_val_i16 (src, kid)); break;
        case GGUF_TYPE_UINT32:  gguf_set_val_u32 (dst, key, gguf_get_val_u32 (src, kid)); break;
        case GGUF_TYPE_INT32:   gguf_set_val_i32 (dst, key, gguf_get_val_i32 (src, kid)); break;
        case GGUF_TYPE_FLOAT32: gguf_set_val_f32 (dst, key, gguf_get_val_f32 (src, kid)); break;
        case GGUF_TYPE_UINT64:  gguf_set_val_u64 (dst, key, gguf_get_val_u64 (src, kid)); break;
        case GGUF_TYPE_INT64:   gguf_set_val_i64 (dst, key, gguf_get_val_i64 (src, kid)); break;
        case GGUF_TYPE_FLOAT64: gguf_set_val_f64 (dst, key, gguf_get_val_f64 (src, kid)); break;
        case GGUF_TYPE_BOOL:    gguf_set_val_bool(dst, key, gguf_get_val_bool(src, kid)); break;
        case GGUF_TYPE_STRING:  gguf_set_val_str (dst, key, gguf_get_val_str (src, kid)); break;
        case GGUF_TYPE_ARRAY:
            {
                const enum gguf_type at = gguf_get_arr_type(src, kid);
                const int64_t n = gguf_get_arr_n(src, kid);
                if (at == GGUF_TYPE_STRING) {
                    std::vector<const char *> strs(n);
                    for (int64_t i = 0; i < n; ++i) {
                        strs[i] = gguf_get_arr_str(src, kid, i);
                    }
                    gguf_set_arr_str(dst, key, strs.data(), n);
                } else {
                    gguf_set_arr_data(dst, key, at, gguf_get_arr_data(src, kid), n);
                }
            } break;
        default:
            break;
    }
}

void llama_safetensors_loader::load_vocab_kv(struct gguf_context * ctx) {
    if (!fs::exists(vocab_path)) {
        throw std::runtime_error(format(
            "tokenizer file '%s' not found - regenerate it with: python3 convert_hf_to_gguf.py %s --vocab-only --outfile %s",
            vocab_path.c_str(), dir_path.c_str(), vocab_path.c_str()));
    }
    gguf_init_params ip = { /*no_alloc=*/ true, /*ctx=*/ nullptr };
    gguf_context_ptr vctx { gguf_init_from_file(vocab_path.c_str(), ip) };
    if (!vctx) {
        throw std::runtime_error(format("failed to load tokenizer file '%s'", vocab_path.c_str()));
    }
    for (int64_t i = 0; i < gguf_get_n_kv(vctx.get()); ++i) {
        const char * key = gguf_get_key(vctx.get(), i);
        if (strncmp(key, "tokenizer.", 10) == 0) {
            copy_kv(ctx, vctx.get(), i);
        }
    }
}

struct gguf_context * llama_safetensors_loader::build_metadata() {
    gguf_context_ptr ctx { gguf_init_empty() };

    gguf_set_val_str(ctx.get(), "general.architecture", "qwen35");
    if (has_fp8_tensors()) {
        gguf_set_val_u32(ctx.get(), "general.file_type", LLAMA_FTYPE_MOSTLY_F8_E4M3);
    }
    {
        std::string dir = dir_path;
        if (!dir.empty() && dir.back() == '/') {
            dir.pop_back();
        }
        gguf_set_val_str(ctx.get(), "general.name", fs::path(dir).filename().string().c_str());
    }

    gguf_set_val_u32(ctx.get(), "qwen35.context_length",       n_ctx);
    gguf_set_val_u32(ctx.get(), "qwen35.embedding_length",     n_embd);
    gguf_set_val_u32(ctx.get(), "qwen35.block_count",          n_layer + n_layer_nextn);
    gguf_set_val_u32(ctx.get(), "qwen35.feed_forward_length",  n_ff);
    gguf_set_val_u32(ctx.get(), "qwen35.attention.head_count",       n_head);
    gguf_set_val_u32(ctx.get(), "qwen35.attention.head_count_kv",    n_head_kv);
    gguf_set_val_u32(ctx.get(), "qwen35.attention.key_length",       n_embd_head_k);
    gguf_set_val_u32(ctx.get(), "qwen35.attention.value_length",     n_embd_head_k);
    gguf_set_val_f32(ctx.get(), "qwen35.attention.layer_norm_rms_epsilon", rms_eps);
    gguf_set_val_u32(ctx.get(), "qwen35.rope.dimension_count", rope_dim);
    gguf_set_val_f32(ctx.get(), "qwen35.rope.freq_base", rope_freq_base);
    gguf_set_arr_data(ctx.get(), "qwen35.rope.dimension_sections", GGUF_TYPE_INT32, mrope_section, 4);
    gguf_set_val_u32(ctx.get(), "qwen35.ssm.conv_kernel",    ssm_d_conv);
    gguf_set_val_u32(ctx.get(), "qwen35.ssm.state_size",     ssm_d_state);
    gguf_set_val_u32(ctx.get(), "qwen35.ssm.group_count",    ssm_n_group);
    gguf_set_val_u32(ctx.get(), "qwen35.ssm.time_step_rank", ssm_dt_rank);
    gguf_set_val_u32(ctx.get(), "qwen35.ssm.inner_size",     ssm_d_inner);
    gguf_set_val_u32(ctx.get(), "qwen35.full_attention_interval", 4);
    if (n_layer_nextn > 0) {
        gguf_set_val_u32(ctx.get(), "qwen35.nextn_predict_layers", n_layer_nextn);
    }
    gguf_set_val_u32(ctx.get(), "qwen35.vocab_size", n_vocab);

    load_vocab_kv(ctx.get());

    ggml_init_params ip = { ggml_tensor_overhead() * (gguf_map.size() + 8), nullptr, true };
    ggml_context_ptr gctx { ggml_init(ip) };
    if (!gctx) {
        throw std::runtime_error("failed to allocate ggml context for tensor metadata");
    }
    for (const auto & [gname, m] : gguf_map) {
        ggml_tensor * t = ggml_new_tensor(gctx.get(), m.type, m.ne.size(), m.ne.data());
        ggml_set_name(t, gname.c_str());
        gguf_add_tensor(ctx.get(), t);
    }

    return ctx.release();
}

// ---------------------------------------------------------------------------
// tensor data

const uint8_t * llama_safetensors_loader::tensor_data(const st_tensor & t) const {
    const auto it = files.find(t.file);
    if (it == files.end() || it->second->addr == nullptr) {
        throw std::runtime_error(format("file '%s' is not mapped", t.file.c_str()));
    }
    return it->second->addr + t.off;
}

void llama_safetensors_loader::fill_fp8(const st_mapping & m, const uint8_t * w, const uint16_t * s, uint8_t * dst) const {
    const int64_t n_in  = m.ne[0];
    const int64_t n_out = m.ne[1];
    const int64_t ncb = n_in / 128;
    const int64_t ncb_s = (n_in + 127) / 128;

    for (int64_t r = 0; r < n_out; ++r) {
        const int64_t sr = m.row_perm ? (*m.row_perm)[r] : r;
        const uint8_t * wrow = w + sr * n_in;
        const uint16_t * srow = s + (sr / 128) * ncb_s;
        uint8_t * drow = dst + r * ncb * 132;
        for (int64_t cb = 0; cb < ncb; ++cb) {
            // the column permutation preserves 128-column blocks (head_dim == 128)
            const int64_t scb = m.col_perm ? (*m.col_perm)[cb * 128] / 128 : cb;
            const float scale = ggml_bf16_to_fp32(ggml_bf16_t{ srow[scb] });
            memcpy(drow + cb * 132, &scale, 4);
            memcpy(drow + cb * 132 + 4, wrow + scb * 128, 128);
        }
    }
}

// copy/reorder src (bf16) into dst, converting to the output element size
void llama_safetensors_loader::fill_bf16(const st_mapping & m, const uint8_t * src, uint8_t * dst) const {
    int64_t n_elements = 1;
    for (int64_t d : m.ne) {
        n_elements *= d;
    }

    switch (m.transform) {
        case TF_NONE:
            memcpy(dst, src, n_elements * 2);
            return;
        case TF_NORM_P1:
            for (int64_t i = 0; i < n_elements; ++i) {
                uint16_t v;
                memcpy(&v, src + i * 2, 2);
                const ggml_bf16_t b = ggml_fp32_to_bf16(ggml_bf16_to_fp32(ggml_bf16_t{ v }) + 1.0f);
                memcpy(dst + i * 2, &b.bits, 2);
            }
            return;
        case TF_A_LOG: // -exp then reorder (head_dim = 1)
            {
                std::vector<ggml_bf16_t> tmp(n_elements);
                for (int64_t i = 0; i < n_elements; ++i) {
                    uint16_t v;
                    memcpy(&v, src + i * 2, 2);
                    tmp[i] = ggml_fp32_to_bf16(-expf(ggml_bf16_to_fp32(ggml_bf16_t{ v })));
                }
                for (int64_t i = 0; i < n_elements; ++i) {
                    memcpy(dst + i * 2, &tmp[(*m.row_perm)[i]].bits, 2);
                }
            }
            return;
        case TF_REORDER:
            {
                const int64_t n_cols = m.ne[0];
                const int64_t n_rows = m.ne.size() > 1 ? m.ne[1] : 1;
                if (m.row_perm) {
                    if (n_rows == 1) {
                        for (int64_t i = 0; i < n_cols; ++i) {
                            memcpy(dst + i * 2, src + (*m.row_perm)[i] * 2, 2);
                        }
                    } else {
                        for (int64_t r = 0; r < n_rows; ++r) {
                            memcpy(dst + r * n_cols * 2, src + (*m.row_perm)[r] * n_cols * 2, n_cols * 2);
                        }
                    }
                } else if (m.col_perm) {
                    for (int64_t r = 0; r < n_rows; ++r) {
                        const uint8_t * srow = src + r * n_cols * 2;
                        uint8_t * drow = dst + r * n_cols * 2;
                        for (int64_t c = 0; c < n_cols; ++c) {
                            memcpy(drow + c * 2, srow + (*m.col_perm)[c] * 2, 2);
                        }
                    }
                } else {
                    memcpy(dst, src, n_elements * 2);
                }
            }
            return;
        default:
            throw std::runtime_error(format("unhandled bf16 transform %d", m.transform));
    }
}

float llama_safetensors_loader::src_val(const st_mapping & m, const uint8_t * src, int64_t i) const {
    if (m.src_f32) {
        float f;
        memcpy(&f, src + i * 4, 4);
        return f;
    }
    uint16_t v;
    memcpy(&v, src + i * 2, 2);
    return ggml_bf16_to_fp32(ggml_bf16_t{ v });
}

void llama_safetensors_loader::fill_f32(const st_mapping & m, const uint8_t * src, uint8_t * dst) const {
    int64_t n_elements = 1;
    for (int64_t d : m.ne) {
        n_elements *= d;
    }

    // the reference converter computes the arithmetic in the HF dtype (bf16 or
    // f32), so round through it when the source is bf16 to stay byte-exact
    auto round_src = [&](float f) {
        return m.src_f32 ? f : ggml_bf16_to_fp32(ggml_fp32_to_bf16(f));
    };

    switch (m.transform) {
        case TF_NONE:
            for (int64_t i = 0; i < n_elements; ++i) {
                const float f = src_val(m, src, i);
                memcpy(dst + i * 4, &f, 4);
            }
            return;
        case TF_NORM_P1:
            for (int64_t i = 0; i < n_elements; ++i) {
                const float f = round_src(src_val(m, src, i) + 1.0f);
                memcpy(dst + i * 4, &f, 4);
            }
            return;
        case TF_A_LOG: // -exp then reorder (head_dim = 1)
            {
                std::vector<float> tmp(n_elements);
                for (int64_t i = 0; i < n_elements; ++i) {
                    tmp[i] = round_src(-expf(src_val(m, src, i)));
                }
                for (int64_t i = 0; i < n_elements; ++i) {
                    memcpy(dst + i * 4, &tmp[(*m.row_perm)[i]], 4);
                }
            }
            return;
        case TF_REORDER:
            {
                const int64_t n_cols = m.ne[0];
                const int64_t n_rows = m.ne.size() > 1 ? m.ne[1] : 1;
                auto conv = [&](int64_t se, uint8_t * de) {
                    const float f = src_val(m, src, se);
                    memcpy(de, &f, 4);
                };
                if (m.row_perm) {
                    if (n_rows == 1) {
                        for (int64_t i = 0; i < n_cols; ++i) {
                            conv((*m.row_perm)[i], dst + i * 4);
                        }
                    } else {
                        for (int64_t r = 0; r < n_rows; ++r) {
                            for (int64_t c = 0; c < n_cols; ++c) {
                                conv((*m.row_perm)[r] * n_cols + c, dst + (r * n_cols + c) * 4);
                            }
                        }
                    }
                } else if (m.col_perm) {
                    for (int64_t r = 0; r < n_rows; ++r) {
                        for (int64_t c = 0; c < n_cols; ++c) {
                            conv(r * n_cols + (*m.col_perm)[c], dst + (r * n_cols + c) * 4);
                        }
                    }
                } else {
                    for (int64_t i = 0; i < n_elements; ++i) {
                        conv(i, dst + i * 4);
                    }
                }
            }
            return;
        default:
            throw std::runtime_error(format("unhandled f32 transform %d", m.transform));
    }
}

void llama_safetensors_loader::fill_tensor(struct ggml_tensor * tensor) {
    const auto it = gguf_map.find(tensor->name);
    if (it == gguf_map.end()) {
        throw std::runtime_error(format("missing safetensors source for tensor '%s'", tensor->name));
    }
    const st_mapping & m = it->second;

    const auto ht = hf_tensors.find(m.hf);
    if (ht == hf_tensors.end()) {
        throw std::runtime_error(format("tensor '%s': safetensors tensor '%s' not found", tensor->name, m.hf.c_str()));
    }
    const uint8_t * data = tensor_data(ht->second);
    const size_t nbytes = ggml_nbytes(tensor);

    // a null buffer means the tensor data is host memory (only possible in tests)
    const bool is_host = tensor->buffer == nullptr || ggml_backend_buffer_is_host(tensor->buffer);

    // plain bf16 copy: no staging needed
    if (m.transform == 0 && m.type == GGML_TYPE_BF16) {
        const size_t chunk = 64 * MIB;
        for (size_t off = 0; off < nbytes; off += chunk) {
            const size_t sz = std::min(chunk, nbytes - off);
            if (is_host) {
                memcpy((uint8_t *) tensor->data + off, data + off, sz);
            } else {
                ggml_backend_tensor_set(tensor, data + off, off, sz);
            }
        }
        return;
    }

    std::vector<uint8_t> staging(nbytes);
    if (m.type == GGML_TYPE_F8_E4M3) {
        const auto st = hf_tensors.find(m.hf_scale);
        if (st == hf_tensors.end()) {
            throw std::runtime_error(format("tensor '%s': fp8 scale tensor '%s' not found", tensor->name, m.hf_scale.c_str()));
        }
        fill_fp8(m, data, (const uint16_t *) tensor_data(st->second), staging.data());
    } else if (m.type == GGML_TYPE_F32) {
        fill_f32(m, data, staging.data());
    } else {
        fill_bf16(m, data, staging.data());
    }

    if (is_host) {
        memcpy(tensor->data, staging.data(), nbytes);
    } else {
        ggml_backend_tensor_set(tensor, staging.data(), 0, nbytes);
    }
}

void llama_safetensors_loader::set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    ((llama_safetensors_loader *) userdata)->fill_tensor(tensor);
}
