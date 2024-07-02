// By Boshi Yuan

#include "fake-offline/FakeInputGate.h"
#include "fake-offline/FakeAddGate.h"
#include "fake-offline/FakeCircuit.h"
#include "share/Spdz2kShare.h"
#include "share/Mod2PowN.h"
#include "share/IsSpdz2kShare.h"
#include "fake-offline/FakeParty.h"

using namespace std;
using namespace md_ml;

int main() {
    using ShrType = Spdz2kShare32;

    FakeParty<ShrType, 2> party("test");
    FakeCircuit<ShrType, 2> circuit(party);

    // // Test for truncation correctness
    // auto a = circuit.input(0, 1, 1);
    // auto b = circuit.input(0, 1, 1);
    // auto c = circuit.multiplyTrunc(a, b);
    //
    // auto d = circuit.output(c);
    //
    // circuit.addEndpoint(d);
    // circuit.runOffline();

    // // Test for Conv2D correctness
    // const size_t rows = 5;
    // const size_t cols = 5;
    // const size_t in_channels = 3;
    // const size_t out_channels = 1;
    // const size_t kernel_shape = 3;
    // const size_t pad = 1;
    // const size_t stride = 1;
    //
    // const Conv2DOp conv_op = {
    //     .kernel_shape_ = {out_channels, in_channels, kernel_shape, kernel_shape},
    //     .input_shape_ = {in_channels, rows, cols},
    //     .output_shape_ = {
    //         out_channels, (rows + 2 * pad + 1 - kernel_shape) / stride, (cols + 2 * pad + 1 - kernel_shape) / stride
    //     },
    //     .dilations_ = {1, 1},
    //     .pads_ = {pad, pad, pad, pad},
    //     .strides_ = {stride, stride}
    // };
    //
    // auto kernel = circuit.input(
    //     0,
    //     conv_op.compute_kernel_size(),
    //     1);
    // auto input_image = circuit.input(
    //     0,
    //     conv_op.compute_input_size(),
    //     1);
    // auto a = circuit.conv2D(input_image, kernel, conv_op);
    // auto o = circuit.output(a);
    //
    // circuit.addEndpoint(o);
    // circuit.runOffline();

    // Test for Gtz correctness
    auto input_x = circuit.input(0, 10, 1);
    auto a = circuit.gtz(input_x);
    auto o = circuit.output(a);

    circuit.addEndpoint(o);
    circuit.runOffline();

    return 0;
}
