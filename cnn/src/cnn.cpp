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

    // Convolution 1

    static float L1[kChannels1][kInSize];
    constexpr int pad = kKernel1 / 2;

    for (int oc = 0; oc < kChannels1; ++oc) {
        for (int x = 0; x < kInSize; ++x) {
            // bias init
            float acc = conv1_bias[oc];

            // 1‑D conv window
            for (int k = 0; k < kKernel1; ++k) {
                int idx = x + k - pad;
                float in_val = 0.f;
                if (idx >= 0 && idx < kInSize) {
                    in_val = input[idx];
                }
                // weight layout: [oc][k] flattened as oc*kKernel1 + k
                acc += in_val * conv1_weight(oc, k);
            }

            L1[oc][x] = acc;
        }
    }

    // Batch Normalization 1
    //   y = (x – running_mean) / sqrt(running_var + eps) * weight + bias

    constexpr float eps = 1e-5f;  // PyTorch’s default eps

    for (int oc = 0; oc < kChannels1; ++oc) {
        // Precompute 1/σ for this channel
        float inv_sigma = 1.0f / std::sqrt(bn1_running_var[oc] + eps);

        // Apply to each position
        for (int x = 0; x < kInSize; ++x) {
            float normalized = (L1[oc][x] - bn1_running_mean[oc]) * inv_sigma;
            L1[oc][x] = normalized * bn1_weight[oc] + bn1_bias[oc];
        }
    }

    // ReLU

    for (int oc = 0; oc < kChannels1; ++oc)
        for (int x = 0; x < kInSize; ++x)
            L1[oc][x] = max(L1[oc][x], 0.0f);

    // Max Pooling 1

    static float P1[kChannels1][kSize2];

    for (int oc = 0; oc < kChannels1; ++oc) {
        for (int i = 0; i < kSize2; ++i) {
            // each pooled window covers two inputs: at positions (i*2) and (i*2 + 1)
            int base = i * 2;
            P1[oc][i] = max(L1[oc][base], L1[oc][base + 1]);
        }
    }

    // Convolution 2

    constexpr int pad2 = kKernel2 / 2;  // = 5/2 = 2
    static float L2[kChannels2][kSize2];

    for (int oc = 0; oc < kChannels2; ++oc) {
        for (int x = 0; x < kSize2; ++x) {
            // start with bias for this output channel
            float acc = conv2_bias[oc];

            // sum over all input channels and kernel positions
            for (int ic = 0; ic < kChannels1; ++ic) {
                for (int k = 0; k < kKernel2; ++k) {
                    int idx = x + k - pad2;
                    float in_val = 0.0f;
                    if (idx >= 0 && idx < kSize2) {
                        // P1 is your pooled output from layer 1
                        in_val = P1[ic][idx];
                    }
                    // flatten conv2_weight as [oc][ic][k]:
                    // index = oc*(kChannels1*kKernel2) + ic*kKernel2 + k
                    acc += in_val * conv2_weight(oc, ic, k);
                }
            }

            // store into L2 for BN2/ReLU2/Pool2 next
            L2[oc][x] = acc;
        }
    }

    // —— BatchNorm2 (inference) on L2[kChannels2][kSize2] ——
    // PyTorch uses eps = 1e-5 by default
    constexpr float eps = 1e-5f;
    for (int oc = 0; oc < kChannels2; ++oc) {
        float inv_sigma = 1.0f / std::sqrt(bn2_running_var[oc] + eps);
        for (int x = 0; x < kSize2; ++x) {
            float normalized = (L2[oc][x] - bn2_running_mean[oc]) * inv_sigma;
            L2[oc][x] = normalized * bn2_weight[oc] + bn2_bias[oc];
        }
    }

    // —— ReLU2 —— 
    for (int oc = 0; oc < kChannels2; ++oc) {
        for (int x = 0; x < kSize2; ++x) {
            L2[oc][x] = max(L2[oc][x], 0.0f);
        }
    }

    // —— MaxPool2 (kernel=2, stride=2) → P2[kChannels2][kSize3] ——
    // kSize3 should be 10 (i.e. floor(20/2))
    static float P2[kChannels2][kSize3];
    for (int oc = 0; oc < kChannels2; ++oc) {
        for (int i = 0; i < kSize3; ++i) {
            int base = i * 2;
            P2[oc][i] = max(L2[oc][base], L2[oc][base + 1]);
        }
    }

    constexpr int pad3 = kKernel3 / 2;  // = 3/2 = 1
    static float L3[kChannels3][kSize3];

    for (int oc = 0; oc < kChannels3; ++oc) {
        for (int x = 0; x < kSize3; ++x) {
            // start with bias for this output channel
            float acc = conv3_bias[oc];

            // sum over all input channels and kernel positions
            for (int ic = 0; ic < kChannels2; ++ic) {
                for (int k = 0; k < kKernel3; ++k) {
                    int idx = x + k - pad3;
                    float in_val = 0.0f;
                    if (idx >= 0 && idx < kSize3) {
                        in_val = P2[ic][idx];
                    }
                    // flatten conv3_weight as [oc][ic][k]
                    acc += in_val * conv3_weight[
                        oc * (kChannels2 * kKernel3)
                    + ic * kKernel3
                    + k
                    ];
                }
            }

            L3[oc][x] = acc;
        }
    }

    // —— BatchNorm3 (inference) ——
    // PyTorch uses eps = 1e-5 by default
    constexpr float eps = 1e-5f;
    for (int oc = 0; oc < kChannels3; ++oc) {
        float inv_sigma = 1.0f / std::sqrt(bn3_running_var[oc] + eps);
        for (int x = 0; x < kSize3; ++x) {
            float normalized = (L3[oc][x] - bn3_running_mean[oc]) * inv_sigma;
            L3[oc][x] = normalized * bn3_weight[oc] + bn3_bias[oc];
        }
    }

    // —— ReLU3 —— 
    for (int oc = 0; oc < kChannels3; ++oc) {
        for (int x = 0; x < kSize3; ++x) {
            L3[oc][x] = max(L3[oc][x], 0.0f);
        }
    }

    // ——— Flatten L3[kChannels3][kSize3] → flat3[LinearSize1] ———
    static float flat3[LinearSize1];
    for (int oc = 0; oc < kChannels3; ++oc) {
        for (int x = 0; x < kSize3; ++x) {
            flat3[oc * kSize3 + x] = L3[oc][x];
        }
    }

    // ——— Fully‑Connected Layer 1: (640 → 128), then ReLU ———
    static float L4[LinearSize2];
    for (int o = 0; o < LinearSize2; ++o) {
        // start with bias
        float acc = fc1_bias[o];
        // dot product
        for (int i = 0; i < LinearSize1; ++i) {
            acc += flat3[i] * fc1_weight[o * LinearSize1 + i];
        }
        // ReLU (dropout skipped at inference)
        L4[o] = acc > 0.0f ? acc : 0.0f;
    }

    // ——— Fully‑Connected Layer 2: (128 → 1000) ———
    for (int o = 0; o < kOutSize; ++o) {
        float acc = fc2_bias[o];
        for (int i = 0; i < LinearSize2; ++i) {
            acc += L4[i] * fc2_weight[o * LinearSize2 + i];
        }
        // write raw scores; RMS normalization comes afterward
        output[o] = acc;
    }
}