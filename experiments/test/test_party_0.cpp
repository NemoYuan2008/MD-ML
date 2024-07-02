// By Boshi Yuan

#include <ranges>
#include "share/Spdz2kShare.h"
#include "protocols/Circuit.h"
#include "utils/print_vector.h"
#include "utils/fixed_point.h"

using namespace std;
using namespace md_ml;

int main() {
    using ShrType = Spdz2kShare32;
    using ClearType = ShrType::ClearType;

    PartyWithFakeOffline<ShrType> party(0, 2, 5050, "test");
    Circuit<ShrType> circuit(party);

    // // Tests for truncation correctness
    // auto a = circuit.input(0, 1, 1);
    // auto b = circuit.input(0, 1, 1);
    // auto c = circuit.multiplyTrunc(a, b);
    // auto d = circuit.output(c);
    // circuit.addEndPoint(d);
    //
    // vector<ClearType> vec{};
    // vec.push_back(double2fix<ClearType>(1.5));
    // a->setInput(vec);
    // b->setInput(vec);
    //
    // circuit.readOfflineFromFile();
    // circuit.runOnlineWithBenckmark();
    //
    // auto o = d->getClear();
    // cout << "output: " << fix2double<ClearType>(o[0]) << endl;

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
    // auto range = views::iota(ClearType(1)) | views::take(conv_op.compute_kernel_size());
    // vector<ClearType> kernel_vec;
    // ranges::copy(range, back_inserter(kernel_vec));
    //
    // range = views::iota(ClearType(1)) | views::take(conv_op.compute_input_size());
    // vector<ClearType> input_image_vec;
    // ranges::copy(range, back_inserter(input_image_vec));
    //
    // PrintVector(kernel_vec);
    // PrintVector(input_image_vec);
    //
    // auto output_vec = convolution(input_image_vec, kernel_vec, conv_op);
    // PrintVector(output_vec);
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
    // kernel->setInput(kernel_vec);
    // input_image->setInput(input_image_vec);
    //
    // circuit.addEndPoint(o);
    //
    // circuit.readOfflineFromFile();
    // circuit.runOnlineWithBenckmark();
    // circuit.printStats();
    //
    // auto output = o->getClear();
    // PrintVector(output);

    // Test for Gtz correctness
    auto input_x = circuit.input(0, 10, 1);
    auto a = circuit.gtz(input_x);
    auto o = circuit.output(a);

    vector<ClearType> input_vec{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    input_x->setInput(input_vec);

    circuit.addEndPoint(o);
    circuit.readOfflineFromFile();
    circuit.runOnlineWithBenckmark();
    circuit.printStats();

    auto output = o->getClear();
    cout << "Output: ";
    PrintVector(output);

    return 0;
}
