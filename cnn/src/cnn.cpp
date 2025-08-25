// --- tuning knobs (safe defaults) ---
#ifndef IC_UNROLL
#define IC_UNROLL 4          // try 2/4/8 depending on DSPs/BRAM
#endif
#ifndef K1_UNROLL
#define K1_UNROLL kKernel1   // fully unroll tiny tap loops
#endif
#ifndef K2_UNROLL
#define K2_UNROLL kKernel2
#endif
#ifndef K3_UNROLL
#define K3_UNROLL kKernel3
#endif

#include <cmath>
#include <tapa.h>
#include "cnn.h"

void CnnKernel(
    tapa::mmap<float> input,

    tapa::mmap<float> conv1_bias,
    tapa::mmap<float> conv2_bias,
    tapa::mmap<float> conv3_bias,
    tapa::mmap<float> conv1_weight,
    tapa::mmap<float> conv2_weight,
    tapa::mmap<float> conv3_weight,

    tapa::mmap<float> bn1_bias,
    tapa::mmap<float> bn2_bias,
    tapa::mmap<float> bn3_bias,
    tapa::mmap<float> bn1_weight,
    tapa::mmap<float> bn2_weight,
    tapa::mmap<float> bn3_weight,
    tapa::mmap<float> bn1_running_mean,
    tapa::mmap<float> bn2_running_mean,
    tapa::mmap<float> bn3_running_mean,
    tapa::mmap<float> bn1_running_var,
    tapa::mmap<float> bn2_running_var,
    tapa::mmap<float> bn3_running_var,

    tapa::mmap<float> fc1_bias,
    tapa::mmap<float> fc2_bias,
    tapa::mmap<float> fc1_weight,
    tapa::mmap<float> fc2_weight,
    
    tapa::mmap<float> output) {

    // ------------------------
    // Tiny caches to avoid repeated DRAM reads
    // ------------------------
    float c1_bias[kChannels1], c2_bias[kChannels2], c3_bias[kChannels3];
    float b1_w[kChannels1], b1_b[kChannels1], b1_m[kChannels1], b1_v[kChannels1];
    float b2_w[kChannels2], b2_b[kChannels2], b2_m[kChannels2], b2_v[kChannels2];
    float b3_w[kChannels3], b3_b[kChannels3], b3_m[kChannels3], b3_v[kChannels3];

    for (int oc = 0; oc < kChannels1; ++oc) {
        c1_bias[oc] = conv1_bias[oc];
        b1_w[oc]    = bn1_weight[oc];
        b1_b[oc]    = bn1_bias[oc];
        b1_m[oc]    = bn1_running_mean[oc];
        b1_v[oc]    = bn1_running_var[oc];
    }
    for (int oc = 0; oc < kChannels2; ++oc) {
        c2_bias[oc] = conv2_bias[oc];
        b2_w[oc]    = bn2_weight[oc];
        b2_b[oc]    = bn2_bias[oc];
        b2_m[oc]    = bn2_running_mean[oc];
        b2_v[oc]    = bn2_running_var[oc];
    }
    for (int oc = 0; oc < kChannels3; ++oc) {
        c3_bias[oc] = conv3_bias[oc];
        b3_w[oc]    = bn3_weight[oc];
        b3_b[oc]    = bn3_bias[oc];
        b3_m[oc]    = bn3_running_mean[oc];
        b3_v[oc]    = bn3_running_var[oc];
    }

    // One-time prefetch of input → tiny local buffer
    float in0[kInSize];
    for (int i = 0; i < kInSize; ++i) in0[i] = input[i];

    // ------------------------
    // Conv1
    // ------------------------
    static float L1[kChannels1][kInSize];
    constexpr int pad1 = kKernel1 / 2;

    // Local copy of conv1 weights (small) → allow full tap unroll
    static float w1[kChannels1][kKernel1];
    #pragma HLS ARRAY_PARTITION variable=w1 complete dim=2
    for (int oc = 0; oc < kChannels1; ++oc)
        for (int k = 0; k < kKernel1; ++k)
        w1[oc][k] = conv1_weight(oc, k);

    for (int oc = 0; oc < kChannels1; ++oc) {
        [[tapa::pipeline(1)]]
        for (int x = 0; x < kInSize; ++x) {
        float acc = c1_bias[oc];
    #pragma HLS UNROLL factor=K1_UNROLL
        for (int k = 0; k < kKernel1; ++k) {
            int idx = x + k - pad1;
            float in_val = (idx >= 0 && idx < kInSize) ? in0[idx] : 0.f;
            acc += in_val * w1[oc][k];
        }
        L1[oc][x] = acc;
        }
    }

    // BN1
    constexpr float eps = 1e-5f;
    for (int oc = 0; oc < kChannels1; ++oc) {
        float inv_sigma = 1.0f / std::sqrt(b1_v[oc] + eps);
        [[tapa::pipeline(1)]]
        for (int x = 0; x < kInSize; ++x) {
        float normalized = (L1[oc][x] - b1_m[oc]) * inv_sigma;
        L1[oc][x] = normalized * b1_w[oc] + b1_b[oc];
        }
    }

    // ReLU1
    for (int oc = 0; oc < kChannels1; ++oc) {
        [[tapa::pipeline(1)]]
        for (int x = 0; x < kInSize; ++x)
        L1[oc][x] = max(L1[oc][x], 0.0f);
    }

    // MaxPool1 → P1
    static float P1[kChannels1][kSize2];
    for (int oc = 0; oc < kChannels1; ++oc) {
        [[tapa::pipeline(1)]]
        for (int i = 0; i < kSize2; ++i) {
        int base = i * 2;
        P1[oc][i] = max(L1[oc][base], L1[oc][base + 1]);
        }
    }

    // ------------------------
    // Conv2
    // ------------------------
    static float L2[kChannels2][kSize2];
    constexpr int pad2 = kKernel2 / 2;

    // Bank along IC since we unroll IC
    #pragma HLS ARRAY_PARTITION variable=P1 cyclic factor=IC_UNROLL dim=1

    for (int oc = 0; oc < kChannels2; ++oc) {
        // Per-OC weight tile
        static float w2[kChannels1][kKernel2];
    #pragma HLS ARRAY_PARTITION variable=w2 cyclic factor=IC_UNROLL dim=1
    #pragma HLS ARRAY_PARTITION variable=w2 complete dim=2
        for (int ic = 0; ic < kChannels1; ++ic)
        for (int k = 0; k < kKernel2; ++k)
            w2[ic][k] = conv2_weight(oc, ic, k);

        [[tapa::pipeline(1)]]
        for (int x = 0; x < kSize2; ++x) {
        float acc = c2_bias[oc];
    #pragma HLS UNROLL factor=IC_UNROLL
        for (int ic = 0; ic < kChannels1; ++ic) {
    #pragma HLS UNROLL factor=K2_UNROLL
            for (int k = 0; k < kKernel2; ++k) {
            int idx = x + k - pad2;
            float in_val = (idx >= 0 && idx < kSize2) ? P1[ic][idx] : 0.0f;
            acc += in_val * w2[ic][k];
            }
        }
        L2[oc][x] = acc;
        }
    }

    // BN2
    for (int oc = 0; oc < kChannels2; ++oc) {
        float inv_sigma = 1.0f / std::sqrt(b2_v[oc] + eps);
        [[tapa::pipeline(1)]]
        for (int x = 0; x < kSize2; ++x) {
        float normalized = (L2[oc][x] - b2_m[oc]) * inv_sigma;
        L2[oc][x] = normalized * b2_w[oc] + b2_b[oc];
        }
    }

    // ReLU2
    for (int oc = 0; oc < kChannels2; ++oc) {
        [[tapa::pipeline(1)]]
        for (int x = 0; x < kSize2; ++x)
        L2[oc][x] = max(L2[oc][x], 0.0f);
    }

    // MaxPool2 → P2
    static float P2[kChannels2][kSize3];
    for (int oc = 0; oc < kChannels2; ++oc) {
        [[tapa::pipeline(1)]]
        for (int i = 0; i < kSize3; ++i) {
        int base = i * 2;
        P2[oc][i] = max(L2[oc][base], L2[oc][base + 1]);
        }
    }

    // ------------------------
    // Conv3
    // ------------------------
    static float L3[kChannels3][kSize3];
    constexpr int pad3 = kKernel3 / 2;

    #pragma HLS ARRAY_PARTITION variable=P2 cyclic factor=IC_UNROLL dim=1

    for (int oc = 0; oc < kChannels3; ++oc) {
        static float w3[kChannels2][kKernel3];
    #pragma HLS ARRAY_PARTITION variable=w3 cyclic factor=IC_UNROLL dim=1
    #pragma HLS ARRAY_PARTITION variable=w3 complete dim=2
        for (int ic = 0; ic < kChannels2; ++ic)
        for (int k = 0; k < kKernel3; ++k)
            w3[ic][k] = conv3_weight(oc, ic, k);   // use your macro

        [[tapa::pipeline(1)]]
        for (int x = 0; x < kSize3; ++x) {
        float acc = c3_bias[oc];
    #pragma HLS UNROLL factor=IC_UNROLL
        for (int ic = 0; ic < kChannels2; ++ic) {
    #pragma HLS UNROLL factor=K3_UNROLL
            for (int k = 0; k < kKernel3; ++k) {
            int idx = x + k - pad3;
            float in_val = (idx >= 0 && idx < kSize3) ? P2[ic][idx] : 0.0f;
            acc += in_val * w3[ic][k];
            }
        }
        L3[oc][x] = acc;
        }
    }

    // BN3
    for (int oc = 0; oc < kChannels3; ++oc) {
        float inv_sigma = 1.0f / std::sqrt(b3_v[oc] + eps);
        [[tapa::pipeline(1)]]
        for (int x = 0; x < kSize3; ++x) {
        float normalized = (L3[oc][x] - b3_m[oc]) * inv_sigma;
        L3[oc][x] = normalized * b3_w[oc] + b3_b[oc];
        }
    }

    // ReLU3
    for (int oc = 0; oc < kChannels3; ++oc) {
        [[tapa::pipeline(1)]]
        for (int x = 0; x < kSize3; ++x)
        L3[oc][x] = max(L3[oc][x], 0.0f);
    }

    // Flatten
    static float flat3[LinearSize1];
    #pragma HLS ARRAY_PARTITION variable=flat3 cyclic factor=IC_UNROLL dim=1
    for (int oc = 0; oc < kChannels3; ++oc) {
        [[tapa::pipeline(1)]]
        for (int x = 0; x < kSize3; ++x)
        flat3[oc * kSize3 + x] = L3[oc][x];
    }

    // FC1 (640 → 128) + ReLU
    static float L4[LinearSize2];
    #pragma HLS ARRAY_PARTITION variable=L4 cyclic factor=IC_UNROLL dim=1
    [[tapa::pipeline(1)]]
    for (int o = 0; o < LinearSize2; ++o) {
        float acc = fc1_bias[o];
    #pragma HLS UNROLL factor=IC_UNROLL
        for (int i = 0; i < LinearSize1; ++i)
        acc += flat3[i] * fc1_weight(o, i);
        L4[o] = max(acc, 0.0f);
    }

    // FC2 (128 → 1000)
    [[tapa::pipeline(1)]]
    for (int o = 0; o < kOutSize; ++o) {
        float acc = fc2_bias[o];
    #pragma HLS UNROLL factor=IC_UNROLL
        for (int i = 0; i < LinearSize2; ++i)
        acc += L4[i] * fc2_weight(o, i);
        output[o] = acc;
    }

    // RMS normalize
    float ms = 0.f;
    [[tapa::pipeline(1)]]
    for (int i = 0; i < kOutSize; ++i) ms += output[i] * output[i];
    ms /= kOutSize;
    constexpr float eps2 = 1e-6f;
    float rms = std::sqrt(ms + eps2);
    [[tapa::pipeline(1)]]
    for (int i = 0; i < kOutSize; ++i) output[i] /= rms;
    }