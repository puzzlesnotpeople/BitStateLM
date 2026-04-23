/**
 * BitStateLM C++ Inference Engine v2.0 with 2-bit Packed Weights
 * Zero dependencies, MatMul-free BitNet inference with 4x weight compression
 * 
 * Packing format: 4 ternary weights per byte (2 bits each)
 *   00 -> 0 (zero)
 *   01 -> 1 (+1)  
 *   10 -> -1
 * 
 * Compile: g++ -O3 -std=c++17 -o bitstate bitstate_packed.cpp
 * Run:     ./bitstate bitstate_packed.bin 0.8 200
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <algorithm>
#include <unordered_map>
#include <cstdint>
#include <cstring>
#include <memory>
#include <chrono>

// =============================================================================
// Config & Tensor Types
// =============================================================================

struct Config {
    int vocab_size;
    int n_layer;
    int n_embd;
    int n_head;
    int head_size;
    int ff_mult;
    int seq_len;
    int use_bitnet;
    int tie_weights;
    float dropout;
    int prenorm;
    int use_rmsnorm;
    int use_swiglu;
    bool packed_weights;  // New: 2-bit packed format
    bool int8_embeddings;  // New: INT8 embeddings
    
    void print() const {
        std::cout << "BitStateLM Config:\n";
        std::cout << "  vocab_size=" << vocab_size << ", n_layer=" << n_layer << "\n";
        std::cout << "  n_embd=" << n_embd << ", n_head=" << n_head << "\n";
        std::cout << "  head_size=" << head_size << ", ff_mult=" << ff_mult << "\n";
        std::cout << "  packed_weights=" << (packed_weights ? "yes (2-bit)" : "no (int8)") << "\n";
        std::cout << "  int8_embeddings=" << (int8_embeddings ? "yes" : "no (fp32)") << "\n";
    }
};

template<typename T>
struct Tensor {
    std::vector<T> data;
    std::vector<size_t> shape;
    
    size_t size() const { return data.size(); }
    size_t numel() const { return data.size(); }
    
    T* ptr() { return data.data(); }
    const T* ptr() const { return data.data(); }
    
    T& operator[](size_t i) { return data[i]; }
    const T& operator[](size_t i) const { return data[i]; }
};

using TensorF32 = Tensor<float>;
using TensorI8 = Tensor<int8_t>;
using TensorU8 = Tensor<uint8_t>;

// =============================================================================
// Binary Loader
// =============================================================================

class BinLoader {
    std::ifstream file;
    
public:
    BinLoader(const std::string& path) : file(path, std::ios::binary) {
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open: " + path);
        }
    }
    
    uint32_t read_u32() {
        uint32_t val;
        file.read(reinterpret_cast<char*>(&val), sizeof(val));
        return val;
    }
    
    int32_t read_i32() {
        int32_t val;
        file.read(reinterpret_cast<char*>(&val), sizeof(val));
        return val;
    }
    
    float read_f32() {
        float val;
        file.read(reinterpret_cast<char*>(&val), sizeof(val));
        return val;
    }
    
    Config load_header() {
        uint32_t magic = read_u32();
        if (magic != 0x42495453) { // "BITS"
            throw std::runtime_error("Invalid magic number");
        }
        
        uint32_t version = read_u32();
        bool packed = (version == 0x00010001) || (version == 0x00010002);  // v1.1 or v1.2
        bool int8_emb = (version == 0x00010002);  // v1.2 = INT8 embeddings
        
        uint32_t num_fields = read_u32();
        if (num_fields < 9) {
            throw std::runtime_error("Invalid config size");
        }
        
        Config cfg;
        cfg.vocab_size = read_i32();
        cfg.n_layer = read_i32();
        cfg.n_embd = read_i32();
        cfg.n_head = read_i32();
        cfg.head_size = read_i32();
        cfg.ff_mult = read_i32();
        cfg.seq_len = read_i32();
        cfg.use_bitnet = read_i32();
        cfg.tie_weights = read_i32();
        cfg.dropout = read_f32();
        
        cfg.prenorm = 0;
        cfg.use_rmsnorm = 1;
        cfg.use_swiglu = 1;
        cfg.packed_weights = packed;
        cfg.int8_embeddings = int8_emb;
        
        return cfg;
    }
    
    // Load INT8 embedding with per-token scales
    // Returns FP32 tensor (dequantized on load)
    TensorF32 load_int8_embedding() {
        // Read shape
        uint32_t ndim = read_u32();
        std::vector<size_t> shape(ndim);
        size_t total = 1;
        for (size_t i = 0; i < ndim; i++) {
            shape[i] = read_u32();
            total *= shape[i];
        }
        
        // Read INT8 data
        size_t vocab_size = shape[0];
        size_t embed_dim = shape[1];
        std::vector<int8_t> int8_data(total);
        file.read(reinterpret_cast<char*>(int8_data.data()), total);
        
        // Read scales (1D array, one per token)
        uint32_t scale_ndim = read_u32();  // should be 1
        uint32_t scale_len = read_u32();   // should be vocab_size
        std::vector<float> scales(scale_len);
        file.read(reinterpret_cast<char*>(scales.data()), scale_len * sizeof(float));
        
        // Dequantize to FP32
        TensorF32 t;
        t.shape = shape;
        t.data.resize(total);
        
        for (size_t v = 0; v < vocab_size; v++) {
            float scale = scales[v];
            for (size_t i = 0; i < embed_dim; i++) {
                t.data[v * embed_dim + i] = int8_data[v * embed_dim + i] * scale;
            }
        }
        
        return t;
    }
    
    TensorF32 load_fp32_tensor() {
        TensorF32 t;
        uint32_t ndim = read_u32();
        t.shape.resize(ndim);
        
        size_t total = 1;
        for (size_t i = 0; i < ndim; i++) {
            t.shape[i] = read_u32();
            total *= t.shape[i];
        }
        
        t.data.resize(total);
        file.read(reinterpret_cast<char*>(t.data.data()), total * sizeof(float));
        
        return t;
    }
    
    TensorI8 load_int8_tensor() {
        TensorI8 t;
        uint32_t ndim = read_u32();
        t.shape.resize(ndim);
        
        size_t total = 1;
        for (size_t i = 0; i < ndim; i++) {
            t.shape[i] = read_u32();
            total *= t.shape[i];
        }
        
        t.data.resize(total);
        file.read(reinterpret_cast<char*>(t.data.data()), total * sizeof(int8_t));
        
        return t;
    }
    
    TensorU8 load_packed_tensor() {
        // Packed tensor: shape same as original, but data is 2-bit packed
        TensorU8 t;
        uint32_t ndim = read_u32();
        t.shape.resize(ndim);
        
        size_t total_elements = 1;
        for (size_t i = 0; i < ndim; i++) {
            t.shape[i] = read_u32();
            total_elements *= t.shape[i];
        }
        
        // Calculate packed size: ceil(total_elements / 4)
        size_t packed_size = (total_elements + 3) / 4;
        t.data.resize(packed_size);
        file.read(reinterpret_cast<char*>(t.data.data()), packed_size);
        
        return t;
    }
    
    bool eof() { return file.eof(); }
};

// =============================================================================
// Math Utils
// =============================================================================

inline float rmsnorm(const float* x, float* out, int n, float eps = 1e-6) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) {
        ss += x[i] * x[i];
    }
    float rms = std::sqrt(ss / n + eps);
    for (int i = 0; i < n; i++) {
        out[i] = x[i] / rms;
    }
    return rms;
}

inline float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

// =============================================================================
// UNPACKED (Legacy) MatMul-free BitNet
// =============================================================================

inline void ternary_linear_unpacked(const float* x, const int8_t* W, float scale,
                                     float* y, int out_features, int in_features) {
    for (int i = 0; i < out_features; i++) {
        float sum = 0.0f;
        const int8_t* w_row = W + i * in_features;
        
        for (int j = 0; j < in_features; j++) {
            int8_t w = w_row[j];
            if (w == 1) {
                sum += x[j];
            } else if (w == -1) {
                sum -= x[j];
            }
        }
        y[i] = sum * scale;
    }
}

// =============================================================================
// PACKED 2-bit MatMul-free BitNet
// =============================================================================

/**
 * Unpack and compute ternary linear in one pass.
 * W_packed: 4 weights per byte, 2 bits each
 *   bits 0-1: weight 0
 *   bits 2-3: weight 1  
 *   bits 4-5: weight 2
 *   bits 6-7: weight 3
 * 
 * Encoding: 00=0, 01=+1, 10=-1
 */
inline void ternary_linear_packed(const float* x, const uint8_t* W_packed, float scale,
                                   float* y, int out_features, int in_features) {
    // Initialize output
    for (int i = 0; i < out_features; i++) {
        y[i] = 0.0f;
    }
    
    // For each output feature (row)
    for (int i = 0; i < out_features; i++) {
        float sum = 0.0f;
        const uint8_t* w_row = W_packed + (i * in_features) / 4;
        
        int j = 0;
        int pack_idx = 0;
        
        while (j < in_features) {
            uint8_t packed = w_row[pack_idx++];
            
            // Unpack 4 weights from this byte
            for (int k = 0; k < 4 && j < in_features; k++, j++) {
                uint8_t code = (packed >> (k * 2)) & 0x03;
                
                // Decode and accumulate
                if (code == 1) {        // 01 = +1
                    sum += x[j];
                } else if (code == 2) { // 10 = -1
                    sum -= x[j];
                }
                // code == 0: 00 = 0, skip
            }
        }
        
        y[i] = sum * scale;
    }
}

// Layers
// =============================================================================

struct LinearLayer {
    TensorI8 weight_i8;       // For unpacked (legacy)
    TensorU8 weight_packed;   // For packed (new)
    float scale;
    int out_features;
    int in_features;
    bool is_packed;
    
    // BitLinear internal LayerNorm parameters
    TensorF32 ln_weight;
    TensorF32 ln_bias;
    bool use_ln;
    
    LinearLayer() : scale(1.0f), out_features(0), in_features(0), is_packed(false), use_ln(false) {}
    
    void load_unpacked(TensorI8&& w, int out_f, int in_f, float s) {
        weight_i8 = std::move(w);
        out_features = out_f;
        in_features = in_f;
        scale = s;
        is_packed = false;
    }
    
    void load_packed(TensorU8&& w, int out_f, int in_f, float s) {
        weight_packed = std::move(w);
        out_features = out_f;
        in_features = in_f;
        scale = s;
        is_packed = true;
    }
    
    // Apply internal LayerNorm (BitLinear-style)
    void apply_ln(float* x, int n) const {
        if (!use_ln || ln_weight.size() == 0) return;
        
        float mean = 0.0f;
        for (int i = 0; i < n; i++) mean += x[i];
        mean /= n;
        
        float var = 0.0f;
        for (int i = 0; i < n; i++) var += (x[i] - mean) * (x[i] - mean);
        var /= n;
        
        float inv_std = 1.0f / std::sqrt(var + 1e-5f);
        
        for (int i = 0; i < n; i++) {
            x[i] = (x[i] - mean) * inv_std * ln_weight[i] + ln_bias[i];
        }
    }
    
    void forward(const float* x, float* y) const {
        // Apply internal LayerNorm first (BitLinear style)
        std::vector<float> x_ln(in_features);
        for (int i = 0; i < in_features; i++) x_ln[i] = x[i];
        apply_ln(x_ln.data(), in_features);
        
        // INT8 activation quantization (matches Python BitLinear._infer_forward)
        // clip_val = abs_mean * 2.5, a_scale = clip_val / 127
        float abs_sum = 0.0f;
        for (int i = 0; i < in_features; i++) abs_sum += std::fabs(x_ln[i]);
        float abs_mean = abs_sum / in_features;
        float clip_val = std::max(abs_mean * 2.5f, 1e-8f);
        float a_scale = clip_val / 127.0f;
        
        // Quantize activations to INT8 (store as float for ternary matmul)
        for (int i = 0; i < in_features; i++) {
            float scaled = x_ln[i] / a_scale;
            float clipped = std::max(-127.0f, std::min(127.0f, scaled));
            x_ln[i] = std::round(clipped);  // INT8 value as float
        }
        
        // Ternary matmul: sum = Σ w_ternary * x_i8
        // Dequantize: y = sum * w_scale * a_scale
        float combined_scale = scale * a_scale;
        if (is_packed) {
            ternary_linear_packed(x_ln.data(), weight_packed.ptr(), combined_scale, y, out_features, in_features);
        } else {
            ternary_linear_unpacked(x_ln.data(), weight_i8.ptr(), combined_scale, y, out_features, in_features);
        }
    }
};

struct NormLayer {
    TensorF32 weight;
    TensorF32 bias;
    bool use_rms = false;  // Initialize to avoid undefined behavior
    
    NormLayer() = default;
    
    void forward(const float* x, float* out, int n) const {
        if (use_rms) {
            rmsnorm(x, out, n);
            if (weight.size() > 0) {
                for (int i = 0; i < n; i++) out[i] *= weight[i];
            }
        } else {
            // Full LayerNorm (nn.LayerNorm in Python)
            float mean = 0.0f;
            for (int i = 0; i < n; i++) mean += x[i];
            mean /= n;

            float var = 0.0f;
            for (int i = 0; i < n; i++) {
                float d = x[i] - mean;
                var += d * d;
            }
            float inv_std = 1.0f / std::sqrt(var / n + 1e-5f);

            for (int i = 0; i < n; i++) {
                out[i] = (x[i] - mean) * inv_std;
                if (weight.size() > 0) out[i] *= weight[i];
                if (bias.size()   > 0) out[i] += bias[i];
            }
        }
    }
};

// GroupNorm for WKV output (PyTorch-compatible: num_groups, num_channels)
// Normalizes each of num_groups groups with num_channels/num_groups channels each
struct GroupNorm {
    TensorF32 weight;
    TensorF32 bias;
    int num_groups;
    int num_channels;
    float eps;
    
    GroupNorm() : num_groups(1), num_channels(0), eps(1e-5f) {}
    
    void forward(const float* x, float* out, int n) const {
        // n = num_channels (e.g., 256)
        // num_groups = e.g., 4 heads
        // channels_per_group = n / num_groups
        int channels_per_group = n / num_groups;
        
        for (int g = 0; g < num_groups; g++) {
            int group_start = g * channels_per_group;
            
            // Compute mean for this group
            float mean = 0.0f;
            for (int i = 0; i < channels_per_group; i++) {
                mean += x[group_start + i];
            }
            mean /= channels_per_group;
            
            // Compute variance for this group
            float var = 0.0f;
            for (int i = 0; i < channels_per_group; i++) {
                float diff = x[group_start + i] - mean;
                var += diff * diff;
            }
            var /= channels_per_group;
            
            float inv_std = 1.0f / std::sqrt(var + eps);
            
            // Normalize with learned affine parameters
            for (int i = 0; i < channels_per_group; i++) {
                float normalized = (x[group_start + i] - mean) * inv_std;
                if (weight.size() > 0) normalized *= weight[group_start + i];
                if (bias.size() > 0) normalized += bias[group_start + i];
                out[group_start + i] = normalized;
            }
        }
    }
};

struct TMBlock {
    LinearLayer r_proj;
    LinearLayer k_proj;
    LinearLayer v_proj;
    LinearLayer g_proj;
    LinearLayer o_proj;
    NormLayer ln1;  // Block-level LN before TM (from BitStateBlock.ln1)
    GroupNorm gn;  // GroupNorm after WKV
    
    // Time-mixing parameters (μ)
    TensorF32 mu_r;
    TensorF32 mu_k;
    TensorF32 mu_v;
    TensorF32 mu_g;
    
    // WKV parameters
    TensorF32 log_decay;  // log(w) for numerical stability
    TensorF32 time_first; // u parameter (exp(u) for bonus)
    
    // Previous x for time-mixing (stored in state)
    
    void forward(const float* x, const float* prev_x, float* out,
                 std::vector<float>& state_wkv,  // [n_head, head_size, head_size]
                 int n_embd, int n_head, int head_size) {
        // x уже нормализован снаружи (ln1), используем как есть
        // prev_x — сырой x до нормализации, для time-mixing
        
        // 2. Time-mixing: compute mixed input with μ parameters
        // dx = prev_x - x
        // mixed = x + dx * mu
        std::vector<float> mixed_r(n_embd), mixed_k(n_embd), 
                          mixed_v(n_embd), mixed_g(n_embd);
        for (int i = 0; i < n_embd; i++) {
            float dx = prev_x[i] - x[i];
            mixed_r[i] = x[i] + dx * (mu_r.size() > 0 ? mu_r[i] : 0.0f);
            mixed_k[i] = x[i] + dx * (mu_k.size() > 0 ? mu_k[i] : 0.0f);
            mixed_v[i] = x[i] + dx * (mu_v.size() > 0 ? mu_v[i] : 0.0f);
            mixed_g[i] = x[i] + dx * (mu_g.size() > 0 ? mu_g[i] : 0.0f);
        }
        
        // 3. Projections
        std::vector<float> r(n_embd), k(n_embd), v(n_embd), g(n_embd);
        r_proj.forward(mixed_r.data(), r.data());
        k_proj.forward(mixed_k.data(), k.data());
        v_proj.forward(mixed_v.data(), v.data());
        g_proj.forward(mixed_g.data(), g.data());
        
        // 4. Activations
        // NOTE: r, k, v have NO activations before WKV in Python!
        // log_decay -> w = exp(-exp(log_decay)) happens INSIDE WKV
        // u -> exp(u) happens INSIDE WKV
        // Only g has silu activation (applied after projection, before gate multiply)
        for (int i = 0; i < n_embd; i++) {
            g[i] = silu(g[i]);  // silu(g) for gate only
        }
        
        // 5. WKV-v4: Matrix state with outer product k⊗v
        std::vector<float> wkv_out(n_embd);
        
        for (int h = 0; h < n_head; h++) {
            int base = h * head_size;
            float* state_h = state_wkv.data() + h * head_size * head_size;
            
            const float* rh = r.data() + base;
            const float* kh = k.data() + base;
            const float* vh = v.data() + base;
            const float* lw = log_decay.ptr() + h * head_size;
            const float* uh = time_first.ptr() + h * head_size;
            float* oh = wkv_out.data() + base;
            
            // Precompute w and u values
            std::vector<float> w_val(head_size), u_val(head_size);
            for (int i = 0; i < head_size; i++) {
                w_val[i] = std::exp(-std::exp(lw[i]));
                u_val[i] = std::exp(uh[i]);
            }
            
            // Correct WKV: y[j] = Σ_i r[i] * (state[i,j] + u[i]*k[i]*v[j])
            // State update: state[i,j] = w[i]*state[i,j] + k[i]*v[j]
            for (int j = 0; j < head_size; j++) {
                float y_j = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    float kv = kh[i] * vh[j];  // k[i] * v[j]
                    float& s_ij = state_h[i * head_size + j];
                    y_j += rh[i] * (s_ij + u_val[i] * kv);
                    s_ij = w_val[i] * s_ij + kv;
                }
                oh[j] = y_j;  // gate applied AFTER GroupNorm
            }
        }
        
        
        // 6. GroupNorm (if weights loaded)
        std::vector<float> gn_out(n_embd);
        if (gn.weight.size() > 0) {
            gn.forward(wkv_out.data(), gn_out.data(), n_embd);
        } else {
            std::memcpy(gn_out.data(), wkv_out.data(), n_embd * sizeof(float));
        }
        
        // 7. Apply gate AFTER GroupNorm (matches Python: out = o_proj(gn(y) * g))
        for (int i = 0; i < n_embd; i++) {
            gn_out[i] *= g[i];
        }
        
        // 8. Output projection — return delta only, residual added in model forward
        o_proj.forward(gn_out.data(), out);
        
    }
};

struct CMBlock {
    LinearLayer k_proj;
    LinearLayer r_proj;
    LinearLayer v_proj;
    NormLayer ln2;  // Block-level LN before CM (from BitStateBlock.ln2)
    
    // Time-mixing parameters for ChannelMix
    TensorF32 mu_k;
    TensorF32 mu_r;
    int ff_mult = 2;  // Default, will be set from config
    
    void forward(const float* x, const float* prev_x, float* out, int n_embd) {
        // x уже нормализован снаружи (ln2), используем как есть
        // prev_x — сырой x до нормализации, для time-mixing
        
        // Time-mixing: compute mixed input with μ parameters
        // dx = prev_x - x
        std::vector<float> mixed_k(n_embd), mixed_r(n_embd);
        for (int i = 0; i < n_embd; i++) {
            float dx = prev_x[i] - x[i];
            mixed_k[i] = x[i] + dx * (mu_k.size() > 0 ? mu_k[i] : 0.0f);
            mixed_r[i] = x[i] + dx * (mu_r.size() > 0 ? mu_r[i] : 0.0f);
        }
        
        int expanded = n_embd * ff_mult;  // ff_mult=2 → expanded=512
        std::vector<float> k(expanded);
        std::vector<float> r(n_embd);
        
        k_proj.forward(mixed_k.data(), k.data());
        r_proj.forward(mixed_r.data(), r.data());
        
        
        // k-activation: relu(k)^2 (as in Python)
        for (int i = 0; i < expanded; i++) {
            float relu_k = k[i] > 0.0f ? k[i] : 0.0f;
            k[i] = relu_k * relu_k;
        }
        
        // Output projection FIRST (matches Python: rgate * v_proj(k_act))
        v_proj.forward(k.data(), out);
        
        // r-gate: sigmoid(r) applied AFTER v_proj
        for (int i = 0; i < n_embd; i++) {
            r[i] = 1.0f / (1.0f + std::exp(-r[i]));  // sigmoid
        }
        
        // Apply gate to output: out[i] *= sigmoid(r[i])
        for (int i = 0; i < n_embd; i++) {
            out[i] *= r[i];
        }
        
    }
};

// =============================================================================
// Full Model
// =============================================================================

struct BitStateLMModel {
    Config cfg;
    TensorF32 token_embedding;
    std::vector<TMBlock> tm_blocks;
    std::vector<CMBlock> cm_blocks;
    NormLayer ln_f;  // Final LayerNorm (ln_f in model.py)
    LinearLayer lm_head;
    std::vector<std::vector<float>> wkv_states;
    std::vector<std::vector<float>> tm_prev_x;  // Previous x for TM time-mixing
    std::vector<std::vector<float>> cm_prev_x;  // Previous x for CM time-mixing
    
    void load(const std::string& bin_path) {
        BinLoader loader(bin_path);
        cfg = loader.load_header();
        cfg.print();
        
        std::cout << "\nLoading weights...\n";
        
        if (cfg.int8_embeddings) {
            token_embedding = loader.load_int8_embedding();
            std::cout << "  Token embedding (INT8 loaded): " << token_embedding.shape[0] << " x " << token_embedding.shape[1] << "\n";
        } else {
            token_embedding = loader.load_fp32_tensor();
            std::cout << "  Token embedding (FP32): " << token_embedding.shape[0] << " x " << token_embedding.shape[1] << "\n";
        }
        
        tm_blocks.resize(cfg.n_layer);
        cm_blocks.resize(cfg.n_layer);
        
        for (int layer = 0; layer < cfg.n_layer; layer++) {
            std::cout << "  Layer " << layer << "...\n";
            
            auto& tm = tm_blocks[layer];
            auto& cm = cm_blocks[layer];
            cm.ff_mult = cfg.ff_mult;  // Set from config
            
            // Load TM projections with internal LayerNorms
            auto load_proj_packed = [&](LinearLayer& proj, int out_f, int in_f) {
                TensorU8 w_packed = loader.load_packed_tensor();
                float scale = loader.load_fp32_tensor()[0];
                proj.load_packed(std::move(w_packed), out_f, in_f, scale);
                // Load internal LayerNorm
                proj.ln_weight = loader.load_fp32_tensor();
                proj.ln_bias = loader.load_fp32_tensor();
                proj.use_ln = true;
            };
            
            auto load_proj_unpacked = [&](LinearLayer& proj, int out_f, int in_f) {
                proj.weight_i8 = loader.load_int8_tensor();
                float scale = loader.load_fp32_tensor()[0];
                proj.load_unpacked(std::move(proj.weight_i8), out_f, in_f, scale);
                // Load internal LayerNorm
                proj.ln_weight = loader.load_fp32_tensor();
                proj.ln_bias = loader.load_fp32_tensor();
                proj.use_ln = true;
            };
            
            if (cfg.packed_weights) {
                load_proj_packed(tm.r_proj, cfg.n_embd, cfg.n_embd);
                load_proj_packed(tm.k_proj, cfg.n_embd, cfg.n_embd);
                load_proj_packed(tm.v_proj, cfg.n_embd, cfg.n_embd);
                load_proj_packed(tm.g_proj, cfg.n_embd, cfg.n_embd);
                load_proj_packed(tm.o_proj, cfg.n_embd, cfg.n_embd);
                
                load_proj_packed(cm.k_proj, cfg.n_embd * cfg.ff_mult, cfg.n_embd);
                load_proj_packed(cm.r_proj, cfg.n_embd, cfg.n_embd);
                load_proj_packed(cm.v_proj, cfg.n_embd, cfg.n_embd * cfg.ff_mult);
            } else {
                load_proj_unpacked(tm.r_proj, cfg.n_embd, cfg.n_embd);
                load_proj_unpacked(tm.k_proj, cfg.n_embd, cfg.n_embd);
                load_proj_unpacked(tm.v_proj, cfg.n_embd, cfg.n_embd);
                load_proj_unpacked(tm.g_proj, cfg.n_embd, cfg.n_embd);
                load_proj_unpacked(tm.o_proj, cfg.n_embd, cfg.n_embd);
                
                load_proj_unpacked(cm.k_proj, cfg.n_embd * cfg.ff_mult, cfg.n_embd);
                load_proj_unpacked(cm.r_proj, cfg.n_embd, cfg.n_embd);
                load_proj_unpacked(cm.v_proj, cfg.n_embd, cfg.n_embd * cfg.ff_mult);
            }
            
            // Load block-level LayerNorms: ln1 (before TM), ln2 (before CM)
            tm.ln1.weight = loader.load_fp32_tensor();
            tm.ln1.bias = loader.load_fp32_tensor();
            cm.ln2.weight = loader.load_fp32_tensor();
            cm.ln2.bias = loader.load_fp32_tensor();
            
            // Load WKV parameters: log_decay and u (time_first)
            tm.log_decay = loader.load_fp32_tensor();
            tm.time_first = loader.load_fp32_tensor();
            
            // Load time-mixing parameters (μ) for TM
            tm.mu_r = loader.load_fp32_tensor();
            tm.mu_k = loader.load_fp32_tensor();
            tm.mu_v = loader.load_fp32_tensor();
            tm.mu_g = loader.load_fp32_tensor();
            
            // Load time-mixing parameters (μ) for CM
            cm.mu_k = loader.load_fp32_tensor();
            cm.mu_r = loader.load_fp32_tensor();
            
            // Load GroupNorm weights (optional, may not exist in older exports)
            try {
                tm.gn.weight = loader.load_fp32_tensor();
                tm.gn.bias = loader.load_fp32_tensor();
            } catch (...) {
                // GroupNorm not present, ignore
            }
        }
        
        // Load final LayerNorm (ln_f in model.py)
        ln_f.weight = loader.load_fp32_tensor();
        ln_f.bias = loader.load_fp32_tensor();
        
        if (!cfg.tie_weights) {
            if (cfg.packed_weights) {
                TensorU8 head_packed = loader.load_packed_tensor();
                float head_scale = loader.load_fp32_tensor()[0];
                lm_head.load_packed(std::move(head_packed), cfg.vocab_size, cfg.n_embd, head_scale);
            } else {
                lm_head.weight_i8 = loader.load_int8_tensor();
                float head_scale = loader.load_fp32_tensor()[0];
                lm_head.load_unpacked(std::move(lm_head.weight_i8),
                                      cfg.vocab_size, cfg.n_embd, head_scale);
            }
        }
        
        wkv_states.resize(cfg.n_layer);
        // RWKV-v4: matrix state [n_head, head_size, head_size]
        int state_per_layer = cfg.n_head * cfg.head_size * cfg.head_size;
        for (int l = 0; l < cfg.n_layer; l++) {
            wkv_states[l].resize(state_per_layer, 0.0f);
        }
        
        // Previous x for time-mixing in each layer
        tm_prev_x.resize(cfg.n_layer);
        cm_prev_x.resize(cfg.n_layer);
        for (int l = 0; l < cfg.n_layer; l++) {
            tm_prev_x[l].resize(cfg.n_embd, 0.0f);
            cm_prev_x[l].resize(cfg.n_embd, 0.0f);
        }
        
        std::cout << "Model loaded successfully!\n";
    }
    
    void forward(int token_id, float* logits) {
        std::vector<float> x(cfg.n_embd);
        float* emb = token_embedding.ptr() + token_id * cfg.n_embd;
        for (int i = 0; i < cfg.n_embd; i++) {
            x[i] = emb[i];
        }
        
        for (int layer = 0; layer < cfg.n_layer; layer++) {
            // Apply ln1 for TM input
            std::vector<float> x_ln1(cfg.n_embd);
            tm_blocks[layer].ln1.forward(x.data(), x_ln1.data(), cfg.n_embd);
            
            std::vector<float> tm_out(cfg.n_embd);
            // TM with time-mixing: pass prev_x (raw) and current x_ln1
            tm_blocks[layer].forward(x_ln1.data(), tm_prev_x[layer].data(), tm_out.data(), 
                                     wkv_states[layer], 
                                     cfg.n_embd, cfg.n_head, cfg.head_size);
            // Update TM prev_x (normalized x = ln1(x), matches Python state[0])
            tm_prev_x[layer] = x_ln1;
            
            // Residual: x = x + tm_out
            for (int i = 0; i < cfg.n_embd; i++) x[i] += tm_out[i];
            
            // Apply ln2 for CM input
            std::vector<float> x_ln2(cfg.n_embd);
            cm_blocks[layer].ln2.forward(x.data(), x_ln2.data(), cfg.n_embd);
            
            std::vector<float> cm_out(cfg.n_embd);
            // CM with time-mixing: pass prev_x (raw) and current x_ln2
            cm_blocks[layer].forward(x_ln2.data(), cm_prev_x[layer].data(), cm_out.data(), 
                                     cfg.n_embd);
            // Update CM prev_x (normalized x = ln2(x), matches Python state)
            cm_prev_x[layer] = x_ln2;
            
            // Residual: x = x + cm_out
            for (int i = 0; i < cfg.n_embd; i++) x[i] += cm_out[i];
        }
        
        // Final LayerNorm (ln_f)
        std::vector<float> x_norm(cfg.n_embd);
        ln_f.forward(x.data(), x_norm.data(), cfg.n_embd);
        
        if (cfg.tie_weights) {
            for (int v = 0; v < cfg.vocab_size; v++) {
                float* w = token_embedding.ptr() + v * cfg.n_embd;
                float sum = 0.0f;
                for (int i = 0; i < cfg.n_embd; i++) {
                    sum += x_norm[i] * w[i];
                }
                logits[v] = sum;
            }
        } else {
            lm_head.forward(x_norm.data(), logits);
        }
    }
    
    // Apply repetition penalty to logits (before softmax)
    void apply_repetition_penalty(float* logits, const std::unordered_map<int, int>& seen_tokens, 
                                   float penalty = 1.1f) {
        for (const auto& [tok_id, count] : seen_tokens) {
            if (tok_id >= 0 && tok_id < cfg.vocab_size) {
                float power = std::min(count, 3);  // cap at 3 like Python
                logits[tok_id] /= std::pow(penalty, power);
            }
        }
    }
    
    int sample(const float* logits, float temperature, int top_k, 
               const std::unordered_map<int, int>& seen_tokens = {}) {
        // Work on a copy since we might apply penalties
        std::vector<float> mod_logits(cfg.vocab_size);
        for (int i = 0; i < cfg.vocab_size; i++) mod_logits[i] = logits[i];
        
        // Apply repetition penalty before temperature
        if (!seen_tokens.empty()) {
            apply_repetition_penalty(mod_logits.data(), seen_tokens, 1.1f);
        }
        
        std::vector<float> probs(cfg.vocab_size);
        
        float max_logit = mod_logits[0];
        for (int i = 1; i < cfg.vocab_size; i++) {
            if (mod_logits[i] > max_logit) max_logit = mod_logits[i];
        }
        
        float sum = 0.0f;
        for (int i = 0; i < cfg.vocab_size; i++) {
            probs[i] = std::exp((mod_logits[i] - max_logit) / temperature);
            sum += probs[i];
        }
        
        for (int i = 0; i < cfg.vocab_size; i++) {
            probs[i] /= sum;
        }
        
        if (top_k > 0 && top_k < cfg.vocab_size) {
            std::vector<std::pair<float, int>> sorted;
            for (int i = 0; i < cfg.vocab_size; i++) {
                sorted.push_back({probs[i], i});
            }
            std::partial_sort(sorted.begin(), sorted.begin() + top_k, sorted.end(),
                             std::greater<std::pair<float, int>>());
            
            std::fill(probs.begin(), probs.end(), 0.0f);
            float topk_sum = 0.0f;
            for (int i = 0; i < top_k; i++) {
                probs[sorted[i].second] = sorted[i].first;
                topk_sum += sorted[i].first;
            }
            for (int i = 0; i < cfg.vocab_size; i++) {
                probs[i] /= topk_sum;
            }
        }
        
        static std::mt19937 gen(42);
        std::discrete_distribution<> dist(probs.begin(), probs.end());
        return dist(gen);
    }
    
    std::vector<int> generate(const std::vector<int>& prompt, int max_new,
                               float temperature = 1.0f) {
        std::vector<float> logits(cfg.vocab_size);
        std::unordered_map<int, int> seen_tokens;  // Track token frequencies for repetition penalty

        // Feed all prompt tokens through model to build up recurrent state
        std::cout << "Processing prompt (" << prompt.size() << " tokens)...\n";
        for (size_t i = 0; i < prompt.size(); i++) {
            forward(prompt[i], logits.data());
            std::cout << "  Token " << prompt[i] << " processed\n";
        }
        
        // DEBUG: Print first 5 logits for verification
        std::cout << "\nC++ LOGITS (first 5): ";
        for (int i = 0; i < 5; i++) {
            std::cout << logits[i] << " ";
        }
        std::cout << "\nARGMAX: " << std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())) << "\n\n";

        std::cout << "Generating " << max_new << " tokens...\n";
        std::vector<int> tokens = prompt;

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < max_new; i++) {
            int next_token = sample(logits.data(), temperature, 50, seen_tokens);
            tokens.push_back(next_token);
            // Update repetition tracking
            seen_tokens[next_token]++;
            forward(next_token, logits.data());

            if ((i + 1) % 10 == 0) {
                auto now = std::chrono::high_resolution_clock::now();
                float elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - t0).count() / 1000.0f;
                if (elapsed > 0)
                    std::cout << "  " << (i+1) << "/" << max_new
                              << "  (" << (i+1)/elapsed << " tok/s)\n";
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        float tok_per_sec = (ms > 0) ? max_new * 1000.0f / ms : 0.0f;
        std::cout << "Done! " << tok_per_sec << " tok/s\n";

        return tokens;
    }
};

// =============================================================================
// Main
// =============================================================================

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "BitStateLM v2.0 - 2-bit Packed Weight Inference\n";
        std::cout << "Usage: " << argv[0] << " <model.bin> [temperature] [max_tokens]\n";
        std::cout << "  temperature: 0.1-2.0 (default: 1.0)\n";
        std::cout << "  max_tokens: 1-1000 (default: 50)\n";
        std::cout << "\nSupports both packed (2-bit) and unpacked (int8) formats\n";
        return 1;
    }
    
    std::string bin_path = argv[1];
    float temperature = (argc > 2) ? std::stof(argv[2]) : 1.0f;
    int max_tokens = (argc > 3) ? std::stoi(argv[3]) : 50;
    
    temperature = std::max(0.1f, std::min(2.0f, temperature));
    max_tokens = std::max(1, std::min(1000, max_tokens));
    
    try {
        BitStateLMModel model;
        model.load(bin_path);
        
        std::cout << "\nBitStateLM C++ Inference Engine\n";
        std::cout << "Temperature: " << temperature << ", Max tokens: " << max_tokens << "\n\n";
        
        std::vector<int> prompt = {1, 2, 3};
        
        auto tokens = model.generate(prompt, max_tokens, temperature);
        
        std::cout << "\nGenerated token IDs: ";
        for (size_t i = prompt.size(); i < tokens.size(); i++) {
            std::cout << tokens[i] << " ";
        }
        std::cout << "\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
