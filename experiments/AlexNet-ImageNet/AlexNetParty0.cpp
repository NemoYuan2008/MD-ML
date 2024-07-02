//
// Created by ysx on 2/6/24.
//
#include <iostream>
#include <filesystem>
#include <vector>

#include "share/Spdz2kShare.h"
#include "protocols/Circuit.h"
#include "protocols/PartyWithFakeOffline.h"
#include "utils/print_vector.h"
#include "utils/fixed_point.h"
#include "AlexNetConfig.h"

using namespace md_ml;

int myid = 0;

std::vector<double> generateRandIn(uint32_t rows, uint32_t cols) {
    std::vector<double> ret(rows * cols);
    for (int i = 0; i < rows * cols; ++i) {
        ret[i] = rand();
    }
    return ret;
}

int main() {
    using ShrType = Spdz2kShare64;
    using ClearType = ShrType::ClearType;

    InitializeConv();

    PartyWithFakeOffline<ShrType> party(0, 2, 5050, "AlexNet-ImageNet");
    Circuit<ShrType> circuit(party);

    std::shared_ptr<Gate<Spdz2kShare64>> out;
    auto start = std::chrono::high_resolution_clock::now();
    auto input_image = circuit.input(0, 3 * rows, cols);
    auto input_x = double2fixVec<Spdz2kShare64::ClearType>(generateRandIn(3 * rows, cols));
    input_image->setInput(input_x);

    out = circuit.addConstant(input_image, 0);

    auto kernel1 = circuit.input(0, Convop[0].kernel_shape_[0] * Convop[0].kernel_shape_[1],
                                 Convop[0].kernel_shape_[2] * Convop[0].kernel_shape_[3]);
    kernel1->setInput(double2fixVec<Spdz2kShare64::ClearType>(generateRandIn(
        Convop[0].kernel_shape_[0] * Convop[0].kernel_shape_[1],
        Convop[0].kernel_shape_[2] * Convop[0].kernel_shape_[3])));

    out = circuit.conv2DTrunc(out, kernel1, Convop[0]);
    out = circuit.relu(out); // relu
    //Pooling1
    out = circuit.avgPool2D(out, Poolop[0]);
    //CL2

    auto kernel2 = circuit.input(0, Convop[1].kernel_shape_[0] * Convop[1].kernel_shape_[1],
                                 Convop[1].kernel_shape_[2] * Convop[1].kernel_shape_[3]);
    kernel2->setInput(double2fixVec<Spdz2kShare64::ClearType>(generateRandIn(
        Convop[1].kernel_shape_[0] * Convop[1].kernel_shape_[1],
        Convop[1].kernel_shape_[2] * Convop[1].kernel_shape_[3])));

    out = circuit.conv2DTrunc(out, kernel2, Convop[1]);
    out = circuit.relu(out); // relu
    out = circuit.avgPool2D(out, Poolop[1]);
    //CL3
    auto kernel3 = circuit.input(0, Convop[2].kernel_shape_[0] * Convop[2].kernel_shape_[1],
                                 Convop[2].kernel_shape_[2] * Convop[2].kernel_shape_[3]);
    kernel3->setInput(double2fixVec<Spdz2kShare64::ClearType>(generateRandIn(
        Convop[2].kernel_shape_[0] * Convop[2].kernel_shape_[1],
        Convop[2].kernel_shape_[2] * Convop[2].kernel_shape_[3])));

    out = circuit.conv2DTrunc(out, kernel3, Convop[2]);
    out = circuit.relu(out); // relu
    //CL4
    auto kernel4 = circuit.input(0, Convop[3].kernel_shape_[0] * Convop[3].kernel_shape_[1],
                                 Convop[3].kernel_shape_[2] * Convop[3].kernel_shape_[3]);

    kernel4->setInput(double2fixVec<Spdz2kShare64::ClearType>(generateRandIn(
        Convop[3].kernel_shape_[0] * Convop[3].kernel_shape_[1],
        Convop[3].kernel_shape_[2] * Convop[3].kernel_shape_[3])));
    //
    out = circuit.conv2DTrunc(out, kernel4, Convop[3]);
    out = circuit.relu(out); // relu
    //CL5
    auto kernel5 = circuit.input(0, Convop[4].kernel_shape_[0] * Convop[4].kernel_shape_[1],
                                 Convop[4].kernel_shape_[2] * Convop[4].kernel_shape_[3]);

    kernel5->setInput(double2fixVec<Spdz2kShare64::ClearType>(generateRandIn(
        Convop[4].kernel_shape_[0] * Convop[4].kernel_shape_[1],
        Convop[4].kernel_shape_[2] * Convop[4].kernel_shape_[3])));

    out = circuit.conv2DTrunc(out, kernel5, Convop[4]);
    out = circuit.relu(out); // relu
    out = circuit.avgPool2D(out, Poolop[2]);
    //FC layer
    auto fc1 = circuit.input(0, fc_n[0], out->dim_col() * out->dim_row());
    fc1->setInput(double2fixVec<Spdz2kShare64::ClearType>(generateRandIn(fc_n[0], out->dim_col() * out->dim_row())));

    out = circuit.multiplyTrunc(fc1, out);
    out = circuit.relu(out);

    auto fc2 = circuit.input(0, fc_n[1], out->dim_col() * out->dim_row());
    fc2->setInput(double2fixVec<Spdz2kShare64::ClearType>(generateRandIn(fc_n[1], out->dim_col() * out->dim_row())));

    out = circuit.multiplyTrunc(fc2, out);
    out = circuit.relu(out);

    auto fc3 = circuit.input(0, fc_n[2], out->dim_col() * out->dim_row());
    fc3->setInput(double2fixVec<Spdz2kShare64::ClearType>(generateRandIn(fc_n[2], out->dim_col() * out->dim_row())));
    out = circuit.multiplyTrunc(fc3, out);
    // softmax?
    //set input
    //    auto input_x = double2fixVec<Spdz2kShare64::ClearType>(generateRandIn(3*rows,cols));
    //    input_image->setInput(input_x);
    //
    //    kernel1->setInput(double2fixVec<Spdz2kShare64::ClearType>(generateRandIn(Convop[0].kernel_shape_[0]*Convop[0].kernel_shape_[1],Convop[0].kernel_shape_[2]*Convop[0].kernel_shape_[3])));
    //    kernel2->setInput(double2fixVec<Spdz2kShare64::ClearType>(generateRandIn(Convop[1].kernel_shape_[0]*Convop[1].kernel_shape_[1],Convop[1].kernel_shape_[2]*Convop[1].kernel_shape_[3])));
    //    kernel3->setInput(double2fixVec<Spdz2kShare64::ClearType>(generateRandIn(Convop[2].kernel_shape_[0]*Convop[2].kernel_shape_[1],Convop[2].kernel_shape_[2]*Convop[2].kernel_shape_[3])));
    //    kernel4->setInput(double2fixVec<Spdz2kShare64::ClearType>(generateRandIn(Convop[3].kernel_shape_[0]*Convop[3].kernel_shape_[1],Convop[3].kernel_shape_[2]*Convop[3].kernel_shape_[3])));
    //    kernel5->setInput(double2fixVec<Spdz2kShare64::ClearType>(generateRandIn(Convop[4].kernel_shape_[0]*Convop[4].kernel_shape_[1],Convop[4].kernel_shape_[2]*Convop[4].kernel_shape_[3])));
    //    fc1->setInput(double2fixVec<Spdz2kShare64::ClearType>(generateRandIn(fc_n[0],out->dim_col() * out->dim_row())));
    //    fc2->setInput(double2fixVec<Spdz2kShare64::ClearType>(generateRandIn(fc_n[1],out->dim_col() * out->dim_row())));
    //    fc3->setInput(double2fixVec<Spdz2kShare64::ClearType>(generateRandIn(fc_n[2],out->dim_col() * out->dim_row())));

    circuit.addEndpoint(out);

    circuit.readOfflineFromFile();
    circuit.runOnlineWithBenckmark();
    circuit.printStats();

    return 0;
}
