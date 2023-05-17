#include "../include/unet.hpp"
#include "../../common/common.hpp"


namespace unet {
    using namespace std;

    inline int upbound(int n, int align = 32) { return (n + align - 1) / align * align; }


    class InferImpl : public Infer {
    public:
        shared_ptr<trt::Infer> trt_;
        string engine_file_;
        Type type_;
        vector<shared_ptr<trt::Memory<unsigned char>>> preprocess_buffers_;
        trt::Memory<float> input_buffer_, predict_;
        int network_input_width_{}, network_input_height_{};
        Norm normalize_;
        int num_classes_ = 0;
        vector<int> network_output_dim;
        bool isdynamic_model_ = false;


        virtual ~InferImpl() = default;


        bool load(const string &engine_file, Type type) {
            trt_ = trt::load(engine_file);
            if (trt_ == nullptr) return false;
            trt_->print();

            isdynamic_model_ = trt_->has_dynamic_dim();
            this->type_ = type;
            auto input_dim = trt_->static_dims(0);
            network_input_width_ = input_dim[3];
            network_input_height_ = input_dim[2];

            network_output_dim = trt_->static_dims(1);
            num_classes_ = network_output_dim[3];

            normalize_ = Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::SwapRB);

            return true;
        }

        void adjust_memory(int batch_size) {
            // the inference batch_size
            size_t input_numel = network_input_width_ * network_input_height_ * 3;
            input_buffer_.gpu(batch_size * input_numel);
            predict_.gpu(batch_size * network_output_dim[1] * network_output_dim[2] * network_output_dim[3]);

            if ((int) preprocess_buffers_.size() < batch_size) {
                for (int i = preprocess_buffers_.size(); i < batch_size; ++i)
                    preprocess_buffers_.push_back(make_shared<trt::Memory<unsigned char>>());
            }
        }

        void preprocess(int ibatch, const cv::Mat &image,
                        shared_ptr<trt::Memory<unsigned char>> preprocess_buffer, AffineMatrix &affine,
                        void *stream = nullptr) {
            affine.compute(make_tuple(image.cols, image.rows),
                           make_tuple(network_input_width_, network_input_height_));

            size_t input_numel = network_input_width_ * network_input_height_ * 3;
            float *input_device = input_buffer_.gpu() + ibatch * input_numel;
            size_t size_image = image.cols * image.rows * 3;
            size_t size_matrix = upbound(sizeof(affine.d2i), 32);
            uint8_t *gpu_workspace = preprocess_buffer->gpu(size_matrix + size_image);
            auto *affine_matrix_device = (float *) gpu_workspace;
            uint8_t *image_device = gpu_workspace + size_matrix;

            uint8_t *cpu_workspace = preprocess_buffer->cpu(size_matrix + size_image);
            auto *affine_matrix_host = (float *) cpu_workspace;
            uint8_t *image_host = cpu_workspace + size_matrix;

            // speed up
            auto stream_ = (cudaStream_t) stream;
            memcpy(image_host, image.data, size_image);
            memcpy(affine_matrix_host, affine.d2i, sizeof(affine.d2i));
            checkRuntime(
                    cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream_));
            checkRuntime(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(affine.d2i),
                                         cudaMemcpyHostToDevice, stream_));

            warp_affine_bilinear_and_normalize_plane(image_device, image.cols * 3, image.cols,
                                                     image.rows, input_device, network_input_width_,
                                                     network_input_height_, affine_matrix_device, 114,
                                                     normalize_, stream_);
        }


        Result forward(const cv::Mat &image, void *stream = nullptr) override {
            auto output = forwards({image}, stream);
            if (output.empty()) return {};
            return output[0];
        }

        vector<Result> forwards(const vector<cv::Mat> &images, void *stream = nullptr) override {
            int num_image = images.size();
            if (num_image == 0) return {};

            auto input_dims = trt_->static_dims(0);
            int infer_batch_size = input_dims[0];
            if (infer_batch_size != num_image) {
                if (isdynamic_model_) {
                    infer_batch_size = num_image;
                    input_dims[0] = num_image;
                    if (!trt_->set_run_dims(0, input_dims)) return {};
                } else {
                    if (infer_batch_size < num_image) {
                        INFO(
                                "When using static shape model, number of images[%d] must be "
                                "less than or equal to the maximum batch[%d].",
                                num_image, infer_batch_size);
                        return {};
                    }
                }
            }

            adjust_memory(num_image);

            vector<AffineMatrix> affine_matrixs(num_image);
            auto stream_ = (cudaStream_t) stream;
            for (int i = 0; i < num_image; ++i)
                preprocess(i, images[i], preprocess_buffers_[i], affine_matrixs[i], stream);

            float *output_device = predict_.gpu();
            vector<void *> bindings{input_buffer_.gpu(), output_device};

            if (!trt_->forward(bindings, stream)) {
                INFO("Failed to tensorRT forward.");
                return {};
            }

            vector<Result> results;
            float *output_host = nullptr;
            checkRuntime(cudaMallocHost(&output_host,
                                        num_image * sizeof(float) * network_output_dim[1] * network_output_dim[2] *
                                        num_classes_));
            checkRuntime(cudaMemcpy(output_host, output_device,
                                    num_image * sizeof(float) * network_output_dim[1] * network_output_dim[2] *
                                    num_classes_, cudaMemcpyDeviceToHost));
            for (int ibatch = 0; ibatch < num_image; ibatch++) {
                cv::Mat output_prob(network_output_dim[1], network_output_dim[2], CV_32F);
                cv::Mat output_index(network_output_dim[1], network_output_dim[2], CV_8U);

                float *pnet = output_host + ibatch * network_output_dim[1] * network_output_dim[2] * num_classes_;
                auto *prob = output_prob.ptr<float>(0);
                auto *pidx = output_index.ptr<uint8_t>(0);

                for (int k = 0; k < output_prob.cols * output_prob.rows; ++k, pnet += num_classes_, ++prob, ++pidx) {
                    int ic = std::max_element(pnet, pnet + num_classes_) - pnet;  //gpu数据
                    *prob = pnet[ic];
                    *pidx = ic;
                }
                cv::Mat InvAffineMat(2, 3, CV_32F, affine_matrixs[ibatch].d2i);
                cv::warpAffine(output_prob, output_prob, InvAffineMat, images[ibatch].size(), cv::INTER_LINEAR);
                cv::warpAffine(output_index, output_index, InvAffineMat, images[ibatch].size(), cv::INTER_LINEAR);
                Result result = make_tuple(output_prob, output_index);
                results.emplace_back(result);
            }
            return results;
        }
    };

    Infer *loadraw(const std::string &engine_file, Type type) {
        auto *impl = new InferImpl();
        if (!impl->load(engine_file, type)) {
            delete impl;
            impl = nullptr;
        }
        return impl;
    }

    shared_ptr<Infer> load(const string &engine_file, Type type) {
        return std::shared_ptr<InferImpl>(
                (InferImpl *) loadraw(engine_file, type));
    }

    static std::vector<int> _classes_colors = {
            0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
            128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
            64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 12
    };

    void render(cv::Mat &image, const cv::Mat &prob, const cv::Mat &iclass) {

        auto pimage = image.ptr<cv::Vec3b>(0);
        auto pprob = prob.ptr<float>(0);
        auto pclass = iclass.ptr<uint8_t>(0);

        for (int i = 0; i < image.cols * image.rows; ++i, ++pimage, ++pprob, ++pclass) {

            int iclass = *pclass;
            float probability = *pprob;
            auto &pixel = *pimage;
            float foreground = std::min(0.6f + probability * 0.2f, 0.8f);
            float background = 1 - foreground;
            for (int c = 0; c < 3; ++c) {
                auto value = pixel[c] * background + foreground * _classes_colors[iclass * 3 + 2 - c];
                pixel[c] = std::min((int) value, 255);
            }
        }
    }
}