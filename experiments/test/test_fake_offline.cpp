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
    using ShrType = Spdz2kShare64;

    FakeParty<ShrType, 2> party("test");
    FakeCircuit<ShrType, 2> circuit(party);

    // // Tests for truncation correctness
    // auto a = circuit.input(0, 1, 1);
    // auto b = circuit.input(0, 1, 1);
    // auto c = circuit.multiplyTrunc(a, b);
    //
    // auto d = circuit.output(c);
    //
    // circuit.addEndpoint(d);
    // circuit.runOffline();

    // Tests for Conv2D correctness
    const int rows = 5;
    const int cols = 5;
    const int in_channels = 3;
    const int out_channels = 1;
    const int kernel_shape = 3;
    const int pad = 1;
    const int stride = 1;

    const Conv2DOp conv_op = {
        .kernel_shape_ = {out_channels, in_channels, kernel_shape, kernel_shape},
        .input_shape_ = {in_channels, rows, cols},
        .output_shape_ = {
            out_channels, (rows + 2 * pad + 1 - kernel_shape) / stride, (cols + 2 * pad + 1 - kernel_shape) / stride
        },
        .dilations_ = {1, 1},
        .pads_ = {pad, pad, pad, pad},
        .strides_ = {stride, stride}
    };

    auto kernel = circuit.input(
        0,
        conv_op.compute_kernel_size(),
        1);
    auto input_image = circuit.input(
        0,
        conv_op.compute_input_size(),
        1);
    auto a = circuit.conv2D(input_image, kernel, conv_op);
    auto o = circuit.output(a);

    circuit.addEndpoint(o);
    circuit.runOffline();

    return 0;
}
