// By Boshi Yuan

#include <ranges>
#include "share/Spdz2kShare.h"
#include "protocols/Circuit.h"
#include "utils/print_vector.h"
#include "utils/fixed_point.h"


using namespace std;
using namespace md_ml;

int main() {
    using ShrType = Spdz2kShare64;
    using ClearType = ShrType::ClearType;

    PartyWithFakeOffline<ShrType> party(1, 2, 5050, "test");
    Circuit<ShrType> circuit(party);

    // // Tests for truncation correctness
    // auto a = circuit.input(0, 1, 1);
    // auto b = circuit.input(0, 1, 1);
    // auto c = circuit.multiplyTrunc(a, b);
    // auto d = circuit.output(c);
    // circuit.addEndpoint(d);

    // circuit.readOfflineFromFile();
    // circuit.runOnlineWithBenckmark();
    // circuit.printStats();

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
    // circuit.readOfflineFromFile();
    // circuit.runOnlineWithBenckmark();
    // circuit.printStats();
    //
    // auto output = o->getClear();
    // PrintVector(output);

    // // Test for Gtz correctness
    // auto input_x = circuit.input(0, 10, 1);
    // auto a = circuit.gtz(input_x);
    // auto o = circuit.output(a);
    //
    // circuit.addEndpoint(o);
    // circuit.readOfflineFromFile();
    // circuit.runOnlineWithBenckmark();
    // circuit.printStats();
    //
    // auto output = o->getClear();
    // PrintVector(output);

    // // Test for ReLU correctness
    // auto a = circuit.input(0, 3, 1);
    // auto b = circuit.relu(a);
    // auto o = circuit.output(b);
    //
    // circuit.addEndpoint(o);
    // circuit.readOfflineFromFile();
    // circuit.runOnlineWithBenckmark();
    // circuit.printStats();
    //
    // auto output = o->getClear();
    // PrintVector(output);

    // Test for AvgPool2D correctness
    const int rows = 32;
    const int cols = 32;
    const int kernel_size = 5;
    const int stride = 1;

    const MaxPoolOp op = {
        .input_shape_ = {1, rows, cols},
        .output_shape_ = {1, (rows + 1 - kernel_size) / stride, (cols + 1 - kernel_size) / stride},
        .kernel_shape_ = {kernel_size, kernel_size},
        .strides_ = {1, 1},
    };

    auto x = circuit.input(0, rows, cols);
    auto a = circuit.avgPool2D(x, op);
    auto o = circuit.output(a);

    circuit.addEndpoint(o);
    circuit.readOfflineFromFile();
    circuit.runOnlineWithBenckmark();
    circuit.printStats();

    auto output = o->getClear();
    cout << "Output: ";
    PrintVector(output);


    return 0;
}
