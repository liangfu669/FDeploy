#include "check.hpp"
#include <tuple>
#include <initializer_list>

enum class NormType : int {
    None = 0, MeanStd = 1, AlphaBeta = 2
};

enum class ChannelType : int {
    None = 0, SwapRB = 1
};

/* 归一化操作，可以支持均值标准差，alpha beta，和swap RB */
struct Norm {
    float mean[3]{};
    float std[3]{};
    float alpha{}, beta{};
    NormType type = NormType::None;
    ChannelType channel_type = ChannelType::None;

    // out = (x * alpha - mean) / std
    static Norm mean_std(const float mean[3], const float std[3], float alpha = 1 / 255.0f,
                         ChannelType channel_type = ChannelType::None) {
        Norm out;
        out.type = NormType::MeanStd;
        out.alpha = alpha;
        out.channel_type = channel_type;
        memcpy(out.mean, mean, sizeof(out.mean));
        memcpy(out.std, std, sizeof(out.std));
        return out;
    }

    // out = x * alpha + beta
    static Norm alpha_beta(float alpha, float beta = 0, ChannelType channel_type = ChannelType::None) {
        Norm out;
        out.type = NormType::AlphaBeta;
        out.alpha = alpha;
        out.beta = beta;
        out.channel_type = channel_type;
        return out;
    }

    // None
    static Norm None() { return {}; }
};


struct AffineMatrix {
    float i2d[6];  // image to dst(network), 2x3 matrix
    float d2i[6];  // dst to image, 2x3 matrix

    void compute(const std::tuple<int, int> &from, const std::tuple<int, int> &to);
};

void warp_affine_bilinear_and_normalize_plane(uint8_t *src, int src_line_size, int src_width,
                                                     int src_height, float *dst, int dst_width,
                                                     int dst_height, float *matrix_2_3,
                                                     uint8_t const_value, const Norm &norm,
                                                     cudaStream_t stream);