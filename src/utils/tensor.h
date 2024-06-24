// The contents of this file is taken from the MOTION2NX project:
// https://github.com/encryptogroup/MOTION2NX/tree/motion2nx/src/motioncore/tensor
// and is licensed under the MIT License.

// I left the code unchanged, but I will refactor it later.

#ifndef MD_ML_TENSOR_H
#define MD_ML_TENSOR_H


#include <vector>
#include <array>
#include <utility>
#include <cassert>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>


struct TensorDimensions {
    std::size_t batch_size_;
    std::size_t num_channels_;
    std::size_t height_;
    std::size_t width_;

    std::size_t get_num_dimensions() const noexcept { return 4; }

    std::size_t get_data_size() const noexcept {
        return batch_size_ * num_channels_ * height_ * width_;
    }

    bool operator==(const TensorDimensions& o) const noexcept {
        return batch_size_ == o.batch_size_ && num_channels_ == o.num_channels_ &&
            height_ == o.height_ && width_ == o.width_;
    }

    bool operator!=(const TensorDimensions& o) const noexcept { return !(*this == o); }
};


struct Conv2DOp {
    std::array<std::size_t, 4> kernel_shape_; // {out_channels, in_channels, kernel_height, kernel_width}
    std::array<std::size_t, 3> input_shape_; // {in_channels, input_height, input_width}
    std::array<std::size_t, 3> output_shape_; // {out_channels, output_height, output_width}

    std::array<std::size_t, 2> dilations_;
    std::array<std::size_t, 4> pads_;
    std::array<std::size_t, 2> strides_;

    bool verify() const noexcept;

    std::array<std::size_t, 3> compute_output_shape() const noexcept;

    std::size_t compute_output_size() const noexcept;

    std::size_t compute_input_size() const noexcept;

    std::size_t compute_kernel_size() const noexcept;

    std::size_t compute_bias_size() const noexcept;

    std::pair<std::size_t, std::size_t> compute_input_matrix_shape() const noexcept;

    std::pair<std::size_t, std::size_t> compute_kernel_matrix_shape() const noexcept;

    std::pair<std::size_t, std::size_t> compute_output_matrix_shape() const noexcept;

    TensorDimensions get_input_tensor_dims() const noexcept;

    TensorDimensions get_kernel_tensor_dims() const noexcept;

    TensorDimensions get_output_tensor_dims() const noexcept;

    bool operator==(const Conv2DOp&) const noexcept;
};


bool Conv2DOp::verify() const noexcept {
    bool result = true;
    result = result && (output_shape_ == compute_output_shape());
    result = result && strides_[0] > 0 && strides_[1] > 0;
    // maybe add more checks here
    return result;
}

std::array<std::size_t, 3> Conv2DOp::compute_output_shape() const noexcept {
    const auto compute_output_dimension = [](auto input_size, auto kernel_size, auto padding_begin,
                                             auto padding_end, auto stride) {
        assert(stride != 0);
        return (input_size - kernel_size + padding_begin + padding_end + stride) / stride;
    };

    std::array<std::size_t, 3> output_shape;
    output_shape[0] = kernel_shape_[0];
    output_shape[1] =
        compute_output_dimension(input_shape_[1], kernel_shape_[2], pads_[0], pads_[2], strides_[0]);
    output_shape[2] =
        compute_output_dimension(input_shape_[2], kernel_shape_[3], pads_[1], pads_[3], strides_[1]);
    return output_shape;
}

std::size_t Conv2DOp::compute_output_size() const noexcept {
    assert(verify());
    auto output_shape = compute_output_shape();
    return output_shape[0] * output_shape[1] * output_shape[2];
}

std::size_t Conv2DOp::compute_input_size() const noexcept {
    assert(verify());
    return input_shape_[0] * input_shape_[1] * input_shape_[2];
}

std::size_t Conv2DOp::compute_kernel_size() const noexcept {
    assert(verify());
    return kernel_shape_[0] * kernel_shape_[1] * kernel_shape_[2] * kernel_shape_[3];
}

std::size_t Conv2DOp::compute_bias_size() const noexcept {
    assert(verify());
    return kernel_shape_[0];
}

std::pair<std::size_t, std::size_t> Conv2DOp::compute_input_matrix_shape() const noexcept {
    assert(verify());
    std::size_t num_rows = kernel_shape_[1] * kernel_shape_[2] * kernel_shape_[3];
    std::size_t num_columns = output_shape_[1] * output_shape_[2];
    return {num_rows, num_columns};
}

std::pair<std::size_t, std::size_t> Conv2DOp::compute_kernel_matrix_shape() const noexcept {
    assert(verify());
    std::size_t num_rows = kernel_shape_[0];
    std::size_t num_columns = kernel_shape_[1] * kernel_shape_[2] * kernel_shape_[3];
    return {num_rows, num_columns};
}

std::pair<std::size_t, std::size_t> Conv2DOp::compute_output_matrix_shape() const noexcept {
    assert(verify());
    std::size_t num_rows = kernel_shape_[0];
    std::size_t num_columns = output_shape_[1] * output_shape_[2];
    return {num_rows, num_columns};
}

TensorDimensions Conv2DOp::get_input_tensor_dims() const noexcept {
    assert(verify());
    return {
        .batch_size_ = 1,
        .num_channels_ = input_shape_[0],
        .height_ = input_shape_[1],
        .width_ = input_shape_[2]
    };
}

TensorDimensions Conv2DOp::get_kernel_tensor_dims() const noexcept {
    assert(verify());
    return {
        .batch_size_ = kernel_shape_[0],
        .num_channels_ = kernel_shape_[1],
        .height_ = kernel_shape_[2],
        .width_ = kernel_shape_[3]
    };
}

TensorDimensions Conv2DOp::get_output_tensor_dims() const noexcept {
    assert(verify());
    return {
        .batch_size_ = 1,
        .num_channels_ = output_shape_[0],
        .height_ = output_shape_[1],
        .width_ = output_shape_[2]
    };
}

bool Conv2DOp::operator==(const Conv2DOp& other) const noexcept {
    assert(verify());
    assert(other.verify());
    bool result = true;
    result = result && kernel_shape_ == other.kernel_shape_;
    result = result && input_shape_ == other.input_shape_;
    result = result && output_shape_ == other.output_shape_;
    result = result && dilations_ == other.dilations_;
    result = result && pads_ == other.pads_;
    result = result && strides_ == other.strides_;
    return result;
}


struct MaxPoolOp {
    std::array<std::size_t, 3> input_shape_;
    std::array<std::size_t, 3> output_shape_;

    std::array<std::size_t, 2> kernel_shape_;
    std::array<std::size_t, 2> strides_;

    bool verify() const noexcept;
    std::array<std::size_t, 3> compute_output_shape() const noexcept;
    std::size_t compute_kernel_size() const noexcept;
    std::size_t compute_input_size() const noexcept;
    std::size_t compute_output_size() const noexcept;
    TensorDimensions get_input_tensor_dims() const noexcept;
    TensorDimensions get_output_tensor_dims() const noexcept;
};


bool MaxPoolOp::verify() const noexcept {
    bool result = true;
    result = result && (output_shape_ == compute_output_shape());
    result = result && strides_[0] > 0 && strides_[1] > 0;
    result = kernel_shape_[0] <= input_shape_[1] && kernel_shape_[1] <= input_shape_[2];
    // maybe add more checks here
    return result;
}

std::array<std::size_t, 3> MaxPoolOp::compute_output_shape() const noexcept {
    const auto compute_output_dimension = [](auto input_size, auto kernel_size, auto stride) {
        assert(stride != 0);
        return (input_size - kernel_size + stride) / stride;
    };

    std::array<std::size_t, 3> output_shape;
    output_shape[0] = input_shape_[0];
    output_shape[1] =
        compute_output_dimension(input_shape_[1], kernel_shape_[2], strides_[0]);
    output_shape[2] =
        compute_output_dimension(input_shape_[2], kernel_shape_[3], strides_[1]);
    return output_shape;
}

std::size_t MaxPoolOp::compute_kernel_size() const noexcept {
    assert(verify());
    return kernel_shape_[0] * kernel_shape_[1];
}

std::size_t MaxPoolOp::compute_input_size() const noexcept {
    assert(verify());
    return input_shape_[0] * input_shape_[1] * input_shape_[2];
}

std::size_t MaxPoolOp::compute_output_size() const noexcept {
    assert(verify());
    return output_shape_[0] * output_shape_[1] * output_shape_[2];
}

TensorDimensions MaxPoolOp::get_input_tensor_dims() const noexcept {
    assert(verify());
    return {
        .batch_size_ = 1,
        .num_channels_ = input_shape_[0],
        .height_ = input_shape_[1],
        .width_ = input_shape_[2]
    };
}

TensorDimensions MaxPoolOp::get_output_tensor_dims() const noexcept {
    assert(verify());
    return {
        .batch_size_ = 1,
        .num_channels_ = output_shape_[0],
        .height_ = output_shape_[1],
        .width_ = output_shape_[2]
    };
}


template <typename T>
void convolution(const T* input_buffer, const T* kernel_buffer, T* output_buffer, const Conv2DOp& conv_op) {
    using TensorType3 = Eigen::Tensor<T, 3, Eigen::RowMajor>;
    using CTensorType3 = Eigen::Tensor<const T, 3, Eigen::RowMajor>;
    using CTensorType4 = Eigen::Tensor<const T, 4, Eigen::RowMajor>;

    assert(conv_op.verify());
    const auto& output_shape = conv_op.output_shape_;
    const auto& input_shape = conv_op.input_shape_;
    const auto& kernel_shape = conv_op.kernel_shape_;

    Eigen::TensorMap<CTensorType3> input(input_buffer, input_shape[0], input_shape[1],
                                         input_shape[2]);
    Eigen::TensorMap<CTensorType4> kernel(kernel_buffer, kernel_shape[0], kernel_shape[1],
                                          kernel_shape[2], kernel_shape[3]);
    Eigen::TensorMap<TensorType3> output(output_buffer, output_shape[0], output_shape[1],
                                         output_shape[2]);
    const std::array<Eigen::Index, 2> kernel_matrix_dimensions = {
        static_cast<Eigen::Index>(kernel_shape[1] * kernel_shape[2] * kernel_shape[3]),
        static_cast<Eigen::Index>(kernel_shape[0])
    };
    const std::array<Eigen::Index, 2> input_matrix_dimensions = {
        static_cast<Eigen::Index>(output_shape[1] * output_shape[2]),
        static_cast<Eigen::Index>(kernel_shape[1] * kernel_shape[2] * kernel_shape[3])
    };

    auto kernel_matrix =
        kernel.shuffle(std::array<int, 4>{3, 2, 1, 0}).reshape(kernel_matrix_dimensions);

    auto input_matrix =
        input.shuffle(Eigen::array<Eigen::Index, 3>{2, 1, 0})
             .extract_image_patches(kernel_shape[2], kernel_shape[3], conv_op.strides_[0],
                                    conv_op.strides_[1], conv_op.dilations_[0], conv_op.dilations_[1],
                                    1, 1, conv_op.pads_[0], conv_op.pads_[2], conv_op.pads_[1],
                                    conv_op.pads_[3], 0)
             .reshape(input_matrix_dimensions);

    const std::array<Eigen::IndexPair<Eigen::Index>, 1> contraction_dimensions = {
        Eigen::IndexPair<Eigen::Index>(1, 0)
    };
    auto output_matrix =
        kernel_matrix.shuffle(std::array<Eigen::Index, 2>{1, 0})
                     .contract(input_matrix.shuffle(std::array<Eigen::Index, 2>{1, 0}), contraction_dimensions)
                     .shuffle(std::array<Eigen::Index, 2>{1, 0});

    const std::array<Eigen::Index, 3> rev_output_dimensions = {
        output.dimension(2), output.dimension(1), output.dimension(0)
    };
    output =
        output_matrix.reshape(rev_output_dimensions).shuffle(Eigen::array<Eigen::Index, 3>{2, 1, 0});
}

template <typename T>
std::vector<T> convolution(const std::vector<T>& input_buffer,
                           const std::vector<T>& kernel_buffer, const Conv2DOp& conv_op) {
    assert(conv_op.verify());
    assert(input_buffer.size() == conv_op.compute_input_size());
    assert(kernel_buffer.size() == conv_op.compute_kernel_size());
    std::vector<T> output_buffer(conv_op.compute_output_size());
    convolution(input_buffer.data(), kernel_buffer.data(), output_buffer.data(), conv_op);
    return output_buffer;
}


template <typename T>
void sumPool(const T* input, T* output, const MaxPoolOp& op) {
    assert(op.verify());
    using TensorType3C = Eigen::Tensor<const T, 3, Eigen::RowMajor>;
    using TensorType3 = Eigen::Tensor<T, 3, Eigen::RowMajor>;
    const auto in_channels = static_cast<Eigen::Index>(op.input_shape_[0]);
    const auto in_rows = static_cast<Eigen::Index>(op.input_shape_[1]);
    const auto in_columns = static_cast<Eigen::Index>(op.input_shape_[2]);
    const auto out_channels = static_cast<Eigen::Index>(op.output_shape_[0]);
    const auto out_rows = static_cast<Eigen::Index>(op.output_shape_[1]);
    const auto out_columns = static_cast<Eigen::Index>(op.output_shape_[2]);
    const auto kernel_rows = static_cast<Eigen::Index>(op.kernel_shape_[0]);
    const auto kernel_columns = static_cast<Eigen::Index>(op.kernel_shape_[1]);
    const auto stride_rows = static_cast<Eigen::Index>(op.strides_[0]);
    const auto stride_columns = static_cast<Eigen::Index>(op.strides_[1]);

    Eigen::TensorMap<TensorType3C> tensor_src(input, in_channels, in_rows, in_columns);
    Eigen::TensorMap<TensorType3> tensor_dst(output, out_channels, out_rows, out_columns);

    tensor_dst = tensor_src.shuffle(Eigen::array<Eigen::Index, 3>{2, 1, 0})
                           .extract_image_patches(kernel_rows, kernel_columns, stride_rows, stride_columns,
                                                  1, 1, 1, 1, 0, 0, 0, 0, T(0))
                           .sum(Eigen::array<Eigen::Index, 2>{1, 2})
                           .reshape(Eigen::array<Eigen::Index, 3>{out_columns, out_rows, out_channels})
                           .shuffle(Eigen::array<Eigen::Index, 3>{2, 1, 0});
}


template <typename T>
inline
std::vector<T> sumPool(const std::vector<T>& inputBuf, const MaxPoolOp& op) {
    std::vector<T> outputBuf(op.compute_output_size());
    sumPool(inputBuf.data(), outputBuf.data(), op);
    return outputBuf;
}


#endif //MD_ML_TENSOR_H
