//
// Created by ysx on 2/6/24.
//
/*
 * AlexNet:
 * C1: kernel 11*11*3, stride=4 output 55*55*48
 * ReLU
 * pool1(bn): kernel 3*3, stride=2, output 27*27*96
 * C2: kernel 5*5*48 output 27*27*256
 * pool2(bn): output 13*13*256
 * C3: kernel 3*3*256
 * C4: kernel 3*3*192
 * C5: kernel 3*3*192
 * pool2(bn): output 13*13*256
 * FC1: kernel 6*6*256
 * FC2: same
 * FC3: same
 */
#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>

#include "share/Spdz2kShare.h"
#include "fake-offline/FakeParty.h"
#include "fake-offline/FakeCircuit.h"
#include "utils/uint128_io.h"

#include "AlexNetConfig.h"

uint32_t parameter_size = 0;

using namespace md_ml;

int main() {
    using ShrType = Spdz2kShare64;

    auto start = std::chrono::high_resolution_clock::now();

    FakeParty<ShrType, 2> party("AlexNet-ImageNet");
    FakeCircuit<ShrType, 2> circuit(party);
    std::shared_ptr<FakeGate<Spdz2kShare64, 2>> out;

    InitializeConv();
    //CL1
    auto input_image = circuit.input(0, 3 * rows, cols);
    parameter_size += Convop[0].kernel_shape_[0] * Convop[0].kernel_shape_[1] * Convop[0].kernel_shape_[2] *
        Convop[0].kernel_shape_[3];

    out = circuit.addConstant(input_image, 0);
    auto kernel1 = circuit.input(0, Convop[0].kernel_shape_[0] * Convop[0].kernel_shape_[1],
                                 Convop[0].kernel_shape_[2] * Convop[0].kernel_shape_[3]);

    out = circuit.conv2DTrunc(out, kernel1, Convop[0]);
    out = circuit.relu(out); // relu
    //Pooling1
    out = circuit.avgPool2D(out, Poolop[0]);
    //CL2

    auto kernel2 = circuit.input(0, Convop[1].kernel_shape_[0] * Convop[1].kernel_shape_[1],
                                 Convop[1].kernel_shape_[2] * Convop[1].kernel_shape_[3]);

    out = circuit.conv2DTrunc(out, kernel2, Convop[1]);
    out = circuit.relu(out); // relu
    out = circuit.avgPool2D(out, Poolop[1]);
    //CL3
    auto kernel3 = circuit.input(0, Convop[2].kernel_shape_[0] * Convop[2].kernel_shape_[1],
                                 Convop[2].kernel_shape_[2] * Convop[2].kernel_shape_[3]);

    out = circuit.conv2DTrunc(out, kernel3, Convop[2]);
    out = circuit.relu(out); // relu
    //CL4
    auto kernel4 = circuit.input(0, Convop[3].kernel_shape_[0] * Convop[3].kernel_shape_[1],
                                 Convop[3].kernel_shape_[2] * Convop[3].kernel_shape_[3]);

    out = circuit.conv2DTrunc(out, kernel4, Convop[3]);
    out = circuit.relu(out); // relu
    //CL5
    auto kernel5 = circuit.input(0, Convop[4].kernel_shape_[0] * Convop[4].kernel_shape_[1],
                                 Convop[4].kernel_shape_[2] * Convop[4].kernel_shape_[3]);

    out = circuit.conv2DTrunc(out, kernel5, Convop[4]);
    out = circuit.relu(out); // relu
    out = circuit.avgPool2D(out, Poolop[2]);
    //FC layer
    auto fc1 = circuit.input(0, fc_n[0], out->dim_col() * out->dim_row());
    out = circuit.multiplyTrunc(fc1, out);
    out = circuit.relu(out);

    auto fc2 = circuit.input(0, fc_n[1], out->dim_col() * out->dim_row());
    out = circuit.multiplyTrunc(fc2, out);
    out = circuit.relu(out);

    auto fc3 = circuit.input(0, fc_n[2], out->dim_col() * out->dim_row());
    out = circuit.multiplyTrunc(fc3, out);
    // softmax?
    circuit.addEndpoint(out);

    circuit.runOffline();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    std::cout << "running time: " << duration.count() << " s\n";
    std::cout << "number of parameter: " << parameter_size << "\n";

    return 0;
}
