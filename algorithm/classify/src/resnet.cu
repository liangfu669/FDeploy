#include "../include/resnet.hpp"
#include "../../common/common.hpp"


namespace resnet {
    using namespace std;

    inline int upbound(int n, int align = 32) { return (n + align - 1) / align * align; }


    class InferImpl : public Infer {
    public:
        shared_ptr<trt::Infer> trt_;
        string engine_file_;
        vector<shared_ptr<trt::Memory<unsigned char>>> preprocess_buffers_;
        trt::Memory<float> input_buffer_, predict_;
        int network_input_width_{}, network_input_height_{};
        Norm normalize_;
        int num_classes_ = 0;
        vector<int> network_output_dim;
        bool isdynamic_model_ = false;


        virtual ~InferImpl() = default;


        bool load(const string &engine_file) {
            trt_ = trt::load(engine_file);
            if (trt_ == nullptr) return false;
            trt_->print();

            isdynamic_model_ = trt_->has_dynamic_dim();
            auto input_dim = trt_->static_dims(0);
            network_input_width_ = input_dim[3];
            network_input_height_ = input_dim[2];

            network_output_dim = trt_->static_dims(1);
            num_classes_ = network_output_dim[1];

            normalize_ = Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::SwapRB);

            return true;
        }

        void adjust_memory(int batch_size) {
            // the inference batch_size
            size_t input_numel = network_input_width_ * network_input_height_ * 3;
            input_buffer_.gpu(batch_size * input_numel);
            predict_.gpu(batch_size * num_classes_);

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

            auto *output = new float[num_classes_ * num_image];
            cudaMemcpy(output, output_device, num_classes_ * sizeof(float) * num_image, cudaMemcpyDeviceToHost);

            vector<Result> results;
            for (int ibatch = 0; ibatch < num_image; ++ibatch){
                Result result{0, 0};
                for (int i = 0; i < num_classes_; ++i) {
                    if (std::get<1>(result) < output[i]) {
                        result = std::make_tuple(i, output[i]);
                    }
                }
                results.emplace_back(result);
            }
            return results;

        }
    };

    Infer *loadraw(const std::string &engine_file) {
        auto *impl = new InferImpl();
        if (!impl->load(engine_file)) {
            delete impl;
            impl = nullptr;
        }
        return impl;
    }

    shared_ptr<Infer> load(const string &engine_file) {
        return std::shared_ptr<InferImpl>(
                (InferImpl *) loadraw(engine_file));
    }

    std::vector<std::string> read_class_name(const std::string &path) {
        std::vector<std::string> class_names;
        std::ifstream infile(path);
        std::string line;
        while (std::getline(infile, line)) {
            class_names.push_back(line);
        }
        return class_names;
    }
}