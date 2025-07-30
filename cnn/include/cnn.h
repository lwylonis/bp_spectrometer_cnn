#ifndef CNN_H_
#define CNN_H_

#include <string>
#include <tapa.h>

using std::string;

template <typename T>
using aligned_vector = std::vector<T, tapa::aligned_allocator<T>>;


#define conv1_weight(o, k) (conv1_weight[ (o) * kKernel1 + (k) ])
#define conv2_weight(o, i, k) (conv2_weight[ (o) * kChannels1 * kKernel2 + (i) * kKernel2 + (k) ])
#define conv3_weight(o, i, k) (conv3_weight[ (o) * kChannels2 * kKernel3 + (i) * kKernel3 + (k) ])

#define fc1_weight(o, i) (fc1_weight[ (o) * LinearSize1 + (i) ])
#define fc2_weight(o, i) (fc2_weight[ (o) * LinearSize2 + (i) ])

#define max(a, b) ((a) > (b) ? (a) : (b))

//MY CONSTANTS: ----------------------------------------
const int kInSize = 41;

// x = self.conv1(x)

const int kChannels1 = 16;
const int kKernel1 = 7;

// x = self.bn1(x)
// x = F.relu(x)
// x = self.pool(x)

const int kSize2 = 20;

// x = self.conv2(x)

const int kChannels2 = 32;
const int kKernel2 = 5;

// x = self.bn2(x)
// x = F.relu(x)
// x = self.pool(x)

const int kSize3 = 10;

// x = self.conv3(x)

const int kChannels3 = 64;
const int kKernel3 = 3;

// x = self.bn3(x)
// x = F.relu(x)
// x = self.dropout(x)
// x = x.flatten(1)

const int LinearSize1 = 640;

// x = self.fc1(x)

const int LinearSize2 = 128;

// x = F.relu(x)
// x = self.dropout(x)
// x = self.fc2(x)
// rms...

const int kOutSize = 1000;
//END MY CONSTANTS: --------------------------------------


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
    
    tapa::mmap<float> output);

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

    aligned_vector<float> & output);

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
    aligned_vector<float> & fc2_weight);

int Verify(const string& data_dir,
           aligned_vector<float> & output);

#endif