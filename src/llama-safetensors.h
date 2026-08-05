// Direct loading of vanilla HuggingFace safetensors models, no GGUF conversion.
//
// The loader parses config.json + model.safetensors.index.json + the per-file
// headers, mmaps the weight files, synthesizes an in-memory gguf_context with
// the model hyperparameters and tensor list, and serves tensor data through
// llama_model_init_from_user().
//
// Only the qwen35 architecture is supported for now (see the tensor name map
// in llama-safetensors.cpp).

#pragma once

#include "ggml.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

struct gguf_context;
struct ggml_context;
struct ggml_tensor;
struct llama_file;
struct llama_mmap;

class llama_safetensors_loader {
public:
    // true if the path is a safetensors model (a dir with config.json + *.safetensors, or a *.safetensors file)
    static bool is_safetensors_path(const char * path);
    // true if at least one registered device can run F8_E4M3 mul_mat natively
    static bool has_fp8_device();

    // throws std::runtime_error on any parse/mmap failure
    llama_safetensors_loader(const char * path);
    ~llama_safetensors_loader();

    llama_safetensors_loader(const llama_safetensors_loader &) = delete;
    llama_safetensors_loader & operator=(const llama_safetensors_loader &) = delete;

    // builds the synthesized gguf metadata context; the caller frees it
    struct gguf_context * build_metadata();

    // true if the model contains F8_E4M3 tensors (drives the hardware gate)
    bool has_fp8_tensors() const;
    // total decoder blocks including the MTP block
    int64_t n_layer_all() const { return n_layer + n_layer_nextn; }

    // data callback for llama_model_init_from_user()
    static void set_tensor_data(struct ggml_tensor * tensor, void * userdata);

private:
    struct st_file {
        std::string path;    // absolute path of the safetensors file
        const uint8_t * addr = nullptr; // mmap base of the whole file
    };

    // one tensor as described by a safetensors header
    struct st_tensor {
        std::string file;    // file name within the model dir
        size_t off = 0;      // tensor data offset within the file
        size_t nbytes = 0;   // tensor data size
        std::string dtype;   // "F8_E4M3" or "BF16"
        std::vector<int64_t> shape;
    };

    // how a gguf tensor maps back to the safetensors data
    struct st_mapping {
        std::string hf;        // HF tensor name
        std::string hf_scale;  // companion *_scale_inv tensor (fp8 only)
        enum ggml_type type = GGML_TYPE_F32;
        int transform = 0;
        bool src_f32 = false;  // the HF tensor is stored as F32 (not BF16)
        const std::vector<int64_t> * row_perm = nullptr; // V-head reorder over rows
        const std::vector<int64_t> * col_perm = nullptr; // V-head reorder over columns
        std::vector<int64_t> ne; // ggml dims, fastest first
    };

    enum st_transform {
        TF_NONE     = 0, // plain copy
        TF_FP8,         // re-block fp8 bytes + bf16 scale into block_f8_e4m3
        TF_NORM_P1,     // + 1.0 (f32, rounded through bf16)
        TF_A_LOG,       // -exp + V-head reorder (head_dim = 1)
        TF_REORDER,     // V-head row/column reorder (row_perm/col_perm)
    };

    // HF -> GGUF name mapping; fills the mapping (transform, perms, type)
    bool map_weight(const std::string & hf, const st_tensor & t, std::string & gname, st_mapping & m) const;

    std::string dir_path;
    std::string vocab_path;    // <dir>/tokenizer.gguf (required, converter-assisted)

    std::map<std::string, st_tensor> hf_tensors;          // HF name -> tensor
    std::map<std::string, st_mapping> gguf_map;           // gguf name -> mapping
    std::map<std::string, std::unique_ptr<st_file>> files;

    std::vector<std::unique_ptr<llama_file>> ll_files;    // keep the fds alive
    std::vector<std::unique_ptr<llama_mmap>> ll_mmaps;

    // hyperparameters (from config.json, text_config)
    int64_t n_embd = 0, n_layer = 0, n_ff = 0, n_head = 0, n_head_kv = 0, n_ctx = 0;
    int64_t ssm_d_conv = 0, ssm_d_inner = 0, ssm_d_state = 0, ssm_dt_rank = 0, ssm_n_group = 0;
    int64_t n_embd_head_k = 0, n_layer_nextn = 0, n_vocab = 0;
    int64_t rope_dim = 0;
    float rms_eps = 0.0f, rope_freq_base = 0.0f;
    int32_t mrope_section[4] = { 0, 0, 0, 0 };

    // V-head reorder permutations (linear attention has 16 K-heads and 32 V-heads)
    std::vector<int64_t> v_perm_1;    // 32 elements, head_dim = 1
    std::vector<int64_t> v_perm_128;  // 4096 elements, head_dim = 128
    std::vector<int64_t> v_perm_128_v; // 4096 elements offset by the q/k rows (in_proj_qkv)

    void parse_config();
    void parse_files();
    void build_mapping();
    void build_permutations();

    void load_vocab_kv(struct gguf_context * ctx);

    void fill_tensor(struct ggml_tensor * tensor);
    const uint8_t * tensor_data(const st_tensor & t) const;

    void fill_fp8(const st_mapping & m, const uint8_t * w, const uint16_t * s, uint8_t * dst) const;
    void fill_bf16(const st_mapping & m, const uint8_t * src, uint8_t * dst) const;
    void fill_f32(const st_mapping & m, const uint8_t * src, uint8_t * dst) const;
    float src_val(const st_mapping & m, const uint8_t * src, int64_t i) const;
};
