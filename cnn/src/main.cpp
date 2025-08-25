#include <chrono>
#include <iostream>
#include <string>

// #include <gflags/gflags.h>
// #include <cstdlib>
// #include <cstdio>?????

#include "cnn.h"

using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::steady_clock;
using std::clog;
using std::endl;
using std::string;

DEFINE_string(btstm, "", "path to the bitstream file, run csim if empty");
DEFINE_string(dtf, "./data", "data directory, default is ./data");

int main(int argc, char** argv) {

    //host data
    aligned_vector<float> h_input(kInSize);

    aligned_vector<float> h_conv1_bias(kChannels1);
    aligned_vector<float> h_conv2_bias(kChannels2);
    aligned_vector<float> h_conv3_bias(kChannels3);
    aligned_vector<float> h_conv1_weight(kChannels1 * kKernel1);
    aligned_vector<float> h_conv2_weight(kChannels2 * kChannels1 * kKernel2);
    aligned_vector<float> h_conv3_weight(kChannels3 * kChannels2 * kKernel3);

    aligned_vector<float> h_bn1_bias(kChannels1);
    aligned_vector<float> h_bn2_bias(kChannels2);
    aligned_vector<float> h_bn3_bias(kChannels3);
    aligned_vector<float> h_bn1_weight(kChannels1);
    aligned_vector<float> h_bn2_weight(kChannels2);
    aligned_vector<float> h_bn3_weight(kChannels3);
    aligned_vector<float> h_bn1_running_mean(kChannels1);
    aligned_vector<float> h_bn2_running_mean(kChannels2);
    aligned_vector<float> h_bn3_running_mean(kChannels3);
    aligned_vector<float> h_bn1_running_var(kChannels1);
    aligned_vector<float> h_bn2_running_var(kChannels2);
    aligned_vector<float> h_bn3_running_var(kChannels3);

    aligned_vector<float> h_fc1_bias(LinearSize2);
    aligned_vector<float> h_fc2_bias(kOutSize);
    aligned_vector<float> h_fc1_weight(LinearSize1*LinearSize2);
    aligned_vector<float> h_fc2_weight(LinearSize2*kOutSize);

    aligned_vector<float> h_output(kOutSize);

    //a vector on host to store data from FPGA device
    aligned_vector<float> d_output(kOutSize);

    if (argc > 2) {
        clog << "Usage: " << argv[0] << " [data dir]\n";
        return EXIT_FAILURE;
    }

    LoadData(
        FLAGS_dtf,
        h_input,
        h_conv1_bias, h_conv2_bias, h_conv3_bias,
        h_conv1_weight, h_conv2_weight, h_conv3_weight,
        h_bn1_bias, h_bn2_bias, h_bn3_bias,
        h_bn1_weight, h_bn2_weight, h_bn3_weight,
        h_bn1_running_mean, h_bn2_running_mean, h_bn3_running_mean,
        h_bn1_running_var,  h_bn2_running_var,  h_bn3_running_var,
        h_fc1_bias, h_fc2_bias,
        h_fc1_weight, h_fc2_weight
    );

    // CPU reference
    clog << "CNN computation on CPU using CnnSequential\n";
    const auto begin = steady_clock::now();
    CnnSequential(
        h_input,
        h_conv1_bias, h_conv2_bias, h_conv3_bias,
        h_conv1_weight, h_conv2_weight, h_conv3_weight,
        h_bn1_bias, h_bn2_bias, h_bn3_bias,
        h_bn1_weight, h_bn2_weight, h_bn3_weight,
        h_bn1_running_mean, h_bn2_running_mean, h_bn3_running_mean,
        h_bn1_running_var,  h_bn2_running_var,  h_bn3_running_var,
        h_fc1_bias, h_fc2_bias,
        h_fc1_weight, h_fc2_weight,
        h_output
    );
    const auto end = steady_clock::now();
    uint64_t run_time_us = duration_cast<microseconds>(end - begin).count();

    // Compute GFLOPS: count MACs (2 FLOPs each) for conv and FC layers
    double ops = 0;
    ops += double(kChannels1) * double(kInSize) * double(kKernel1) * 2; //conv1
    ops += double(kChannels2) * double(kSize2) * double(kChannels1) * double(kKernel2) * 2; //conv2
    ops += double(kChannels3) * double(kSize3) * double(kChannels2) * double(kKernel3) * 2; //conv3
    ops += double(LinearSize2) * double(LinearSize1) * 2; //fc1
    ops += double(kOutSize) * double(LinearSize2) * 2; //fc2
    float gflops = ops / (run_time_us * 1e3);
    clog << "Time: " << run_time_us * 1e-6 << " s\n";
    clog << "Perf: " << gflops << " GFlops (don't trust if you sw emu hw emu)\n";

    // FPGA kernel invocation
    double time_taken = tapa::invoke(
        CnnKernel, FLAGS_btstm,
        tapa::read_only_mmap<float>(h_input),
        tapa::read_only_mmap<float>(h_conv1_bias),
        tapa::read_only_mmap<float>(h_conv2_bias),
        tapa::read_only_mmap<float>(h_conv3_bias),
        tapa::read_only_mmap<float>(h_conv1_weight),
        tapa::read_only_mmap<float>(h_conv2_weight),
        tapa::read_only_mmap<float>(h_conv3_weight),
        tapa::read_only_mmap<float>(h_bn1_bias),
        tapa::read_only_mmap<float>(h_bn2_bias),
        tapa::read_only_mmap<float>(h_bn3_bias),
        tapa::read_only_mmap<float>(h_bn1_weight),
        tapa::read_only_mmap<float>(h_bn2_weight),
        tapa::read_only_mmap<float>(h_bn3_weight),
        tapa::read_only_mmap<float>(h_bn1_running_mean),
        tapa::read_only_mmap<float>(h_bn2_running_mean),
        tapa::read_only_mmap<float>(h_bn3_running_mean),
        tapa::read_only_mmap<float>(h_bn1_running_var),
        tapa::read_only_mmap<float>(h_bn2_running_var),
        tapa::read_only_mmap<float>(h_bn3_running_var),
        tapa::read_only_mmap<float>(h_fc1_bias),
        tapa::read_only_mmap<float>(h_fc2_bias),
        tapa::read_only_mmap<float>(h_fc1_weight),
        tapa::read_only_mmap<float>(h_fc2_weight),
        tapa::write_only_mmap<float>(d_output)
    );
    time_taken *= 1e-6; // total time in mini second
    printf("Kernel time is %f ms\n", time_taken * 1000);

    // Verification
    int error = Verify(FLAGS_dtf, d_output);
    if (error != 0) {
        clog << "Found " << error << " error" << (error > 1 ? "s\n" : "\n");
        clog << "FAIL" << endl;
        return EXIT_FAILURE;
    } else {
        clog << "PASS" << endl;
        return EXIT_SUCCESS;
    }

}