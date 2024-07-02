//
// Created by ysx on 2/6/24.
//

#ifndef MD_ML_ALEXNETCONFIG_H
#define MD_ML_ALEXNETCONFIG_H
#include "utils/linear_algebra.h"
#include "utils/tensor.h"
// using cifar10 32*32 dataset
const int times = 1;
const int rows = 224;
const int cols = 224;
const int classes = 1000;
const int initchannels = 3;
// ImageNet 224*224*1000

std::vector<int> fc_n = { 9216, 4096, 1000};
std::vector<int> expansion1 = {1,2,2,2};
std::vector<int> expansion2 = {1,1,1,1};
std::vector<uint8_t> stride = {1,1,1,1};
std::vector <Conv2DOp> Convop(5);
std::vector <MaxPoolOp> Poolop(3);
void InitializeConv(){
//    uint32_t conv_in_channels[0] = initchannels; // initial input channels
//    uint32_t conv_out_channels[0];
//    uint32_t conv2_in_channels;
//    uint32_t conv2_out_channels;
//    uint32_t conv3_in_channels;
//    uint32_t conv3_out_channels;
    uint32_t figsize = 224;
    std::vector<uint32_t> conv_in_channels(5), conv_out_channels(5);
    conv_in_channels = {3,96,256,384,384};
    conv_out_channels = {96,256,384,384,256};
    uint32_t parameter_size=0;
    
        //ImageNet 224*224 * 1000
//        conv_in_channels[0] = 3;
//        conv_out_channels[0] = 64;
        Convop[0].kernel_shape_ = {conv_out_channels[0],conv_in_channels[0],11,11};
        Convop[0].input_shape_ = {conv_in_channels[0],224,224};
        Convop[0].output_shape_ = {conv_out_channels[0],55,55};
        Convop[0].dilations_ ={1,1};
        Convop[0].pads_ ={2,2,2,2};
        Convop[0].strides_ ={4,4}; //strides = 4

        uint32_t kernel_size = 3;
        uint32_t poolstride = 2;
        Poolop[0].input_shape_ = {96, 55, 55};
        Poolop[0].output_shape_ = {96, 27, 27};
        Poolop[0].kernel_shape_ = {kernel_size, kernel_size};
        Poolop[0].strides_ = {poolstride, poolstride};
//        std::cout<<"C1 layers:-----------\n";
//        std::cout<< "output: "<<conv_out_channels[0]<<"\n";


    Convop[1].kernel_shape_ = {conv_out_channels[1],conv_in_channels[1],5,5};
    Convop[1].input_shape_ = {conv_in_channels[1],27,27};
    Convop[1].output_shape_ = {conv_out_channels[1],27,27};
    Convop[1].dilations_ ={1,1};
    Convop[1].pads_ ={2,2,2,2};
    Convop[1].strides_ ={1,1};

    kernel_size = 3;
    poolstride = 2;
    Poolop[1].input_shape_ = {256, 27, 27};
    Poolop[1].output_shape_ = {256, 13, 13};
    Poolop[1].kernel_shape_ = {kernel_size, kernel_size};
    Poolop[1].strides_ = {poolstride, poolstride};

    Convop[2].kernel_shape_ = {conv_out_channels[2],conv_in_channels[2],3,3};
    Convop[2].input_shape_ = {conv_in_channels[2],13,13};
    Convop[2].output_shape_ = {conv_out_channels[2],13,13};
    Convop[2].dilations_ ={1,1};
    Convop[2].pads_ ={1,1,1,1};
    Convop[2].strides_ ={1,1};

    Convop[3].kernel_shape_ = {conv_out_channels[3],conv_in_channels[3],3,3};
    Convop[3].input_shape_ = {conv_in_channels[3],13,13};
    Convop[3].output_shape_ = {conv_out_channels[3],13,13};
    Convop[3].dilations_ ={1,1};
    Convop[3].pads_ ={1,1,1,1};
    Convop[3].strides_ ={1,1};

    Convop[4].kernel_shape_ = {conv_out_channels[4],conv_in_channels[4],3,3};
    Convop[4].input_shape_ = {conv_in_channels[4],13,13};
    Convop[4].output_shape_ = {conv_out_channels[4],13,13};
    Convop[4].dilations_ ={1,1};
    Convop[4].pads_ ={1,1,1,1};
    Convop[4].strides_ ={1,1};

    kernel_size = 3;
    poolstride = 2;
    Poolop[2].input_shape_ = {256, 13, 13};
    Poolop[2].output_shape_ = {256, 6, 6};
    Poolop[2].kernel_shape_ = {kernel_size, kernel_size};
    Poolop[2].strides_ = {poolstride, poolstride};
}
const Conv2DOp conv1_op = {.kernel_shape_ = {initchannels, 3, 3, 3},
        .input_shape_ = {3, rows, cols},
        .output_shape_ = {initchannels, rows, cols},
        .dilations_ = {1, 1},
        .pads_ = {1, 1, 1, 1},
        .strides_ = {1, 1}
};

#endif //MD_ML_ALEXNETCONFIG_H
