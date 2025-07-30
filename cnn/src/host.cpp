#include <stdexcept>
#include <string>
#include <fcntl.h>
#include <sys/mman.h>
#include <tapa.h>
#include "cnn.h"

using std::clog;
using std::endl;
using std::string;

// Sequential CNN implementation
void CnnSequential(
    aligned_vector<float> & input,

    aligned_vector<float> & conv1_bias,
    aligned_vector<float> & conv2_bias,
    aligned_vector<float> & conv3_bias,
    aligned_vector<float> & conv1_weight,
    aligned_vector<float> & conv2_weight,
    aligned_vector<float> & conv3_weight,

    aligned_vector<float> & bn1_bias,
    aligned_vector<float> & bn2_bias,
    aligned_vector<float> & bn3_bias,
    aligned_vector<float> & bn1_weight,
    aligned_vector<float> & bn2_weight,
    aligned_vector<float> & bn3_weight,
    aligned_vector<float> & bn1_running_mean,
    aligned_vector<float> & bn2_running_mean,
    aligned_vector<float> & bn3_running_mean,
    aligned_vector<float> & bn1_running_var,
    aligned_vector<float> & bn2_running_var,
    aligned_vector<float> & bn3_running_var,

    aligned_vector<float> & fc1_bias,
    aligned_vector<float> & fc2_bias,
    aligned_vector<float> & fc1_weight,
    aligned_vector<float> & fc2_weight,

    aligned_vector<float> & output) {

    //TODO
    for (int i = 0; i < 1000; ++i) { output[i] = 0; } //TEMPORARY

}

void LoadData(
    const string& data_dir, 
    aligned_vector<float> & input,

    aligned_vector<float> & conv1_bias,
    aligned_vector<float> & conv2_bias,
    aligned_vector<float> & conv3_bias,
    aligned_vector<float> & conv1_weight,
    aligned_vector<float> & conv2_weight,
    aligned_vector<float> & conv3_weight,

    aligned_vector<float> & bn1_bias,
    aligned_vector<float> & bn2_bias,
    aligned_vector<float> & bn3_bias,
    aligned_vector<float> & bn1_weight,
    aligned_vector<float> & bn2_weight,
    aligned_vector<float> & bn3_weight,
    aligned_vector<float> & bn1_running_mean,
    aligned_vector<float> & bn2_running_mean,
    aligned_vector<float> & bn3_running_mean,
    aligned_vector<float> & bn1_running_var,
    aligned_vector<float> & bn2_running_var,
    aligned_vector<float> & bn3_running_var,

    aligned_vector<float> & fc1_bias,
    aligned_vector<float> & fc2_bias,
    aligned_vector<float> & fc1_weight,
    aligned_vector<float> & fc2_weight) {

    // File names
    const char* kInputFile           = "/input.bin";

    const char* kConv1BiasFile       = "/conv1_bias.bin";
    const char* kConv1WeightFile     = "/conv1_weight.bin";
    const char* kConv2BiasFile       = "/conv2_bias.bin";
    const char* kConv2WeightFile     = "/conv2_weight.bin";
    const char* kConv3BiasFile       = "/conv3_bias.bin";
    const char* kConv3WeightFile     = "/conv3_weight.bin";

    const char* kBN1BiasFile         = "/bn1_bias.bin";
    const char* kBN1WeightFile       = "/bn1_weight.bin";
    const char* kBN1MeanFile         = "/bn1_running_mean.bin";
    const char* kBN1VarFile          = "/bn1_running_var.bin";
    const char* kBN2BiasFile         = "/bn2_bias.bin";
    const char* kBN2WeightFile       = "/bn2_weight.bin";
    const char* kBN2MeanFile         = "/bn2_running_mean.bin";
    const char* kBN2VarFile          = "/bn2_running_var.bin";
    const char* kBN3BiasFile         = "/bn3_bias.bin";
    const char* kBN3WeightFile       = "/bn3_weight.bin";
    const char* kBN3MeanFile         = "/bn3_running_mean.bin";
    const char* kBN3VarFile          = "/bn3_running_var.bin";

    const char* kFC1BiasFile         = "/fc1_bias.bin";
    const char* kFC1WeightFile       = "/fc1_weight.bin";
    const char* kFC2BiasFile         = "/fc2_bias.bin";
    const char* kFC2WeightFile       = "/fc2_weight.bin";

    // Helper lambda to open & mmap, then memcpy & cleanup
    auto load_bin = [&](const char* fname, float* dst, size_t count) {
        string path = data_dir + fname;
        int fd = open(path.c_str(), O_RDONLY);
        if (fd == -1) {
            clog << "Cannot find " << path << "\n";
            exit(EXIT_FAILURE);
        }
        size_t nbytes = count * sizeof(float);
        float* src = reinterpret_cast<float*>(
            mmap(nullptr, nbytes, PROT_READ, MAP_SHARED, fd, 0));
        if (src == MAP_FAILED) {
            clog << "Failed to mmap " << path << "\n";
            close(fd);
            exit(EXIT_FAILURE);
        }
        memcpy(dst, src, nbytes);
        munmap(src, nbytes);
        close(fd);
    };

    // Load all arrays
    load_bin(kInputFile,       input.data(),           kInSize);

    load_bin(kConv1BiasFile,   conv1_bias.data(),      kChannels1);
    load_bin(kConv1WeightFile, conv1_weight.data(),    kChannels1 * kKernel1);
    load_bin(kConv2BiasFile,   conv2_bias.data(),      kChannels2);
    load_bin(kConv2WeightFile, conv2_weight.data(),    kChannels2 * kChannels1 * kKernel2);
    load_bin(kConv3BiasFile,   conv3_bias.data(),      kChannels3);
    load_bin(kConv3WeightFile, conv3_weight.data(),    kChannels3 * kChannels2 * kKernel3);

    load_bin(kBN1BiasFile,     bn1_bias.data(),        kChannels1);
    load_bin(kBN1WeightFile,   bn1_weight.data(),      kChannels1);
    load_bin(kBN1MeanFile,     bn1_running_mean.data(),kChannels1);
    load_bin(kBN1VarFile,      bn1_running_var.data(), kChannels1);

    load_bin(kBN2BiasFile,     bn2_bias.data(),        kChannels2);
    load_bin(kBN2WeightFile,   bn2_weight.data(),      kChannels2);
    load_bin(kBN2MeanFile,     bn2_running_mean.data(),kChannels2);
    load_bin(kBN2VarFile,      bn2_running_var.data(), kChannels2);

    load_bin(kBN3BiasFile,     bn3_bias.data(),        kChannels3);
    load_bin(kBN3WeightFile,   bn3_weight.data(),      kChannels3);
    load_bin(kBN3MeanFile,     bn3_running_mean.data(),kChannels3);
    load_bin(kBN3VarFile,      bn3_running_var.data(), kChannels3);

    load_bin(kFC1BiasFile,     fc1_bias.data(),        LinearSize2);
    load_bin(kFC1WeightFile,   fc1_weight.data(),      LinearSize2 * LinearSize1);
    load_bin(kFC2BiasFile,     fc2_bias.data(),        kOutSize);
    load_bin(kFC2WeightFile,   fc2_weight.data(),      kOutSize * LinearSize2);
}

float IsError(float a, float b) {
    return fabs((a - b) / (a + b)) > 1e-3f && fabs(a - b) > 0.05f;
}

int Verify(const string& data_dir,
           aligned_vector<float>& output) {

    int error = 0;
    const char kOutputFile[] = "/output.bin";
    string path = data_dir + kOutputFile;

    // 1) Open ground‑truth file
    int fd = open(path.c_str(), O_RDONLY);
    if (fd == -1) {
        std::clog << "Cannot find " << path << std::endl;
        return EXIT_FAILURE;
    }

    // 2) mmap it (size = kOutSize floats)
    size_t mapped_bytes = sizeof(float) * kOutSize;
    float* ground_truth = reinterpret_cast<float*>(
        mmap(nullptr,
             mapped_bytes,
             PROT_READ,
             MAP_SHARED,
             fd, 0));
    if (ground_truth == MAP_FAILED) {
        std::clog << "Failed to mmap " << path << std::endl;
        close(fd);
        return EXIT_FAILURE;
    }

    // 3) Compare element‑wise
    bool first = true;
    for (int i = 0; i < kOutSize; ++i) {
        if (IsError(output[i], ground_truth[i])) {
            if (first) {
                std::clog << "First error: got " << output[i]
                          << ", expecting " << ground_truth[i]
                          << " @ index " << i << std::endl;
                first = false;
            }
            ++error;
        }
    }

    // 4) Clean up
    munmap(ground_truth, mapped_bytes);
    close(fd);

    return error;
}