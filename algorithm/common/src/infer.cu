
#include <NvInfer.h>

#include <fstream>
#include <numeric>
#include <sstream>
#include <unordered_map>

#include "../include/infer.hpp"
#include "../include/check.hpp"

namespace trt {

    using namespace std;
    using namespace nvinfer1;


    static std::string format_shape(const Dims &shape) {
        stringstream output;
        char buf[64];
        const char *fmts[] = {"%d", "x%d"};
        for (int i = 0; i < shape.nbDims; ++i) {
            snprintf(buf, sizeof(buf), fmts[i != 0], shape.d[i]);
            output << buf;
        }
        return output.str();
    }

    Timer::Timer() {
        checkRuntime(cudaEventCreate((cudaEvent_t *) &start_));
        checkRuntime(cudaEventCreate((cudaEvent_t *) &stop_));
    }

    Timer::~Timer() {
        checkRuntime(cudaEventDestroy((cudaEvent_t) start_));
        checkRuntime(cudaEventDestroy((cudaEvent_t) stop_));
    }

    void Timer::start(void *stream) {
        stream_ = stream;
        checkRuntime(cudaEventRecord((cudaEvent_t) start_, (cudaStream_t) stream_));
    }

    float Timer::stop(const char *prefix, bool print) {
        checkRuntime(cudaEventRecord((cudaEvent_t) stop_, (cudaStream_t) stream_));
        checkRuntime(cudaEventSynchronize((cudaEvent_t) stop_));

        float latency = 0;
        checkRuntime(cudaEventElapsedTime(&latency, (cudaEvent_t) start_, (cudaEvent_t) stop_));

        if (print) {
            printf("[%s]: %.5f ms\n", prefix, latency);
        }
        return latency;
    }

    BaseMemory::BaseMemory(void *cpu, size_t cpu_bytes, void *gpu, size_t gpu_bytes) {
        reference(cpu, cpu_bytes, gpu, gpu_bytes);
    }

    void BaseMemory::reference(void *cpu, size_t cpu_bytes, void *gpu, size_t gpu_bytes) {
        release();

        if (cpu == nullptr || cpu_bytes == 0) {
            cpu = nullptr;
            cpu_bytes = 0;
        }

        if (gpu == nullptr || gpu_bytes == 0) {
            gpu = nullptr;
            gpu_bytes = 0;
        }

        this->cpu_ = cpu;
        this->cpu_capacity_ = cpu_bytes;
        this->cpu_bytes_ = cpu_bytes;
        this->gpu_ = gpu;
        this->gpu_capacity_ = gpu_bytes;
        this->gpu_bytes_ = gpu_bytes;

        this->owner_cpu_ = !(cpu && cpu_bytes > 0);
        this->owner_gpu_ = !(gpu && gpu_bytes > 0);
    }

    BaseMemory::~BaseMemory() { release(); }

    void *BaseMemory::gpu_realloc(size_t bytes) {
        if (gpu_capacity_ < bytes) {
            release_gpu();

            gpu_capacity_ = bytes;
            checkRuntime(cudaMalloc(&gpu_, bytes));
            // checkRuntime(cudaMemset(gpu_, 0, size));
        }
        gpu_bytes_ = bytes;
        return gpu_;
    }

    void *BaseMemory::cpu_realloc(size_t bytes) {
        if (cpu_capacity_ < bytes) {
            release_cpu();

            cpu_capacity_ = bytes;
            checkRuntime(cudaMallocHost(&cpu_, bytes));
            Assert(cpu_ != nullptr);
            // memset(cpu_, 0, size);
        }
        cpu_bytes_ = bytes;
        return cpu_;
    }

    void BaseMemory::release_cpu() {
        if (cpu_) {
            if (owner_cpu_) {
                checkRuntime(cudaFreeHost(cpu_));
            }
            cpu_ = nullptr;
        }
        cpu_capacity_ = 0;
        cpu_bytes_ = 0;
    }

    void BaseMemory::release_gpu() {
        if (gpu_) {
            if (owner_gpu_) {
                checkRuntime(cudaFree(gpu_));
            }
            gpu_ = nullptr;
        }
        gpu_capacity_ = 0;
        gpu_bytes_ = 0;
    }

    void BaseMemory::release() {
        release_cpu();
        release_gpu();
    }

    class _native_nvinfer_logger : public ILogger {
    public:
        void log(Severity severity, const char *msg) noexcept override {
            if (severity == Severity::kINTERNAL_ERROR) {
                INFO("NVInfer INTERNAL_ERROR: %s", msg);
                abort();
            } else if (severity == Severity::kERROR) {
                INFO("NVInfer: %s", msg);
            }
            // else  if (severity == Severity::kWARNING) {
            //     INFO("NVInfer: %s", msg);
            // }
            // else  if (severity == Severity::kINFO) {
            //     INFO("NVInfer: %s", msg);
            // }
            // else {
            //     INFO("%s", msg);
            // }
        }
    };

    static _native_nvinfer_logger gLogger;

    template<typename T>
    static void destroy_nvidia_pointer(T *ptr) {
        if (ptr) ptr->destroy();
    }

    static std::vector<uint8_t> load_file(const string &file) {
        ifstream in(file, ios::in | ios::binary);
        if (!in.is_open()) return {};

        in.seekg(0, ios::end);
        size_t length = in.tellg();

        std::vector<uint8_t> data;
        if (length > 0) {
            in.seekg(0, ios::beg);
            data.resize(length);

            in.read((char *) &data[0], length);
        }
        in.close();
        return data;
    }

    class _native_engine_context {
    public:
        virtual ~_native_engine_context() { destroy(); }

        bool construct(const void *pdata, size_t size) {
            destroy();

            if (pdata == nullptr || size == 0) return false;

            runtime_ = shared_ptr<IRuntime>(createInferRuntime(gLogger), destroy_nvidia_pointer < IRuntime > );
            if (runtime_ == nullptr) return false;

            engine_ = shared_ptr<ICudaEngine>(runtime_->deserializeCudaEngine(pdata, size, nullptr),
                                              destroy_nvidia_pointer < ICudaEngine > );
            if (engine_ == nullptr) return false;

            context_ = shared_ptr<IExecutionContext>(engine_->createExecutionContext(),
                                                     destroy_nvidia_pointer < IExecutionContext > );
            return context_ != nullptr;
        }

    private:
        void destroy() {
            context_.reset();
            engine_.reset();
            runtime_.reset();
        }

    public:
        shared_ptr<IExecutionContext> context_;
        shared_ptr<ICudaEngine> engine_;
        shared_ptr<IRuntime> runtime_ = nullptr;
    };

    class InferImpl : public Infer {
    public:
        shared_ptr<_native_engine_context> context_;
        unordered_map<string, int> binding_name_to_index_;

        virtual ~InferImpl() = default;

        bool construct(const void *data, size_t size) {
            context_ = make_shared<_native_engine_context>();
            if (!context_->construct(data, size)) {
                return false;
            }

            setup();
            return true;
        }

        bool load(const string &file) {
            auto data = load_file(file);
            if (data.empty()) {
                INFO("An empty file has been loaded. Please confirm your file path: %s", file.c_str());
                return false;
            }
            return this->construct(data.data(), data.size());
        }

        void setup() {
            auto engine = this->context_->engine_;
            int nbBindings = engine->getNbBindings();

            binding_name_to_index_.clear();
            for (int i = 0; i < nbBindings; ++i) {
                const char *bindingName = engine->getBindingName(i);
                binding_name_to_index_[bindingName] = i;
            }
        }

        int index(const std::string &name) override {
            auto iter = binding_name_to_index_.find(name);
            Assertf(iter != binding_name_to_index_.end(), "Can not found the binding name: %s",
                    name.c_str());
            return iter->second;
        }

        bool forward(const std::vector<void *> &bindings, void *stream,
                             void *input_consum_event) override {
            return this->context_->context_->enqueueV2((void **) bindings.data(), (cudaStream_t) stream,
                                                       (cudaEvent_t *) input_consum_event);
        }

        std::vector<int> run_dims(const std::string &name) override {
            return run_dims(index(name));
        }

        std::vector<int> run_dims(int ibinding) override {
            auto dim = this->context_->context_->getBindingDimensions(ibinding);
            return std::vector<int>(dim.d, dim.d + dim.nbDims);
        }

        std::vector<int> static_dims(const std::string &name) override {
            return static_dims(index(name));
        }

        std::vector<int> static_dims(int ibinding) override {
            auto dim = this->context_->engine_->getBindingDimensions(ibinding);
            return std::vector<int>(dim.d, dim.d + dim.nbDims);
        }

        int num_bindings() override { return this->context_->engine_->getNbBindings(); }

        bool is_input(int ibinding) override {
            return this->context_->engine_->bindingIsInput(ibinding);
        }

        bool set_run_dims(const std::string &name, const std::vector<int> &dims) override {
            return this->set_run_dims(index(name), dims);
        }

        bool set_run_dims(int ibinding, const std::vector<int> &dims) override {
            Dims d;
            memcpy(d.d, dims.data(), sizeof(int) * dims.size());
            d.nbDims = dims.size();
            return this->context_->context_->setBindingDimensions(ibinding, d);
        }

        int numel(const std::string &name) override { return numel(index(name)); }

        int numel(int ibinding) override {
            auto dim = this->context_->context_->getBindingDimensions(ibinding);
            return std::accumulate(dim.d, dim.d + dim.nbDims, 1, std::multiplies<int>());
        }

        DType dtype(const std::string &name) override { return dtype(index(name)); }

        DType dtype(int ibinding) override {
            return (DType) this->context_->engine_->getBindingDataType(ibinding);
        }

        bool has_dynamic_dim() override {
            // check if any input or output bindings have dynamic shapes
            // code from ChatGPT
            int numBindings = this->context_->engine_->getNbBindings();
            for (int i = 0; i < numBindings; ++i) {
                nvinfer1::Dims dims = this->context_->engine_->getBindingDimensions(i);
                for (int j = 0; j < dims.nbDims; ++j) {
                    if (dims.d[j] == -1) return true;
                }
            }
            return false;
        }

        void print() override {
            INFO("Infer %p [%s]", this, has_dynamic_dim() ? "DynamicShape" : "StaticShape");

            int num_input = 0;
            int num_output = 0;
            auto engine = this->context_->engine_;
            for (int i = 0; i < engine->getNbBindings(); ++i) {
                if (engine->bindingIsInput(i))
                    num_input++;
                else
                    num_output++;
            }

            INFO("Inputs: %d", num_input);
            for (int i = 0; i < num_input; ++i) {
                auto name = engine->getBindingName(i);
                auto dim = engine->getBindingDimensions(i);
                INFO("\t%d.%s : shape {%s}", i, name, format_shape(dim).c_str());
            }

            INFO("Outputs: %d", num_output);
            for (int i = 0; i < num_output; ++i) {
                auto name = engine->getBindingName(i + num_input);
                auto dim = engine->getBindingDimensions(i + num_input);
                INFO("\t%d.%s : shape {%s}", i, name, format_shape(dim).c_str());
            }
        }
    };

    Infer *loadraw(const std::string &file) {
        auto *impl = new InferImpl();
        if (!impl->load(file)) {
            delete impl;
            impl = nullptr;
        }
        return impl;
    }

    std::shared_ptr<Infer> load(const std::string &file) {
        return std::shared_ptr<InferImpl>((InferImpl *) loadraw(file));
    }

    std::string format_shape(const std::vector<int> &shape) {
        stringstream output;
        char buf[64];
        const char *fmts[] = {"%d", "x%d"};
        for (int i = 0; i < (int) shape.size(); ++i) {
            snprintf(buf, sizeof(buf), fmts[i != 0], shape[i]);
            output << buf;
        }
        return output.str();
    }
};  // namespace trt
