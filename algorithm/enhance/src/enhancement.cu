#include "iostream"
#include "opencv2/opencv.hpp"

static std::string file_name(const std::string &path, bool include_suffix) {
    if (path.empty()) return "";

    int p = path.rfind('/');
    int e = path.rfind('\\');
    p = std::max(p, e);
    p += 1;

    // include suffix
    if (include_suffix) return path.substr(p);

    int u = path.rfind('.');
    if (u == -1) return path.substr(p);

    if (u <= p) u = path.size();
    return path.substr(p, u - p);
}

void __log_func(const char *file, int line, const char *fmt, ...) {
    va_list vl;
    va_start(vl, fmt);
    char buffer[2048];
    std::string filename = file_name(file, true);
    int n = snprintf(buffer, sizeof(buffer), "[%s:%d]: ", filename.c_str(), line);
    vsnprintf(buffer + n, sizeof(buffer) - n, fmt, vl);
    fprintf(stdout, "%s\n", buffer);
}

#define INFO(...) __log_func(__FILE__, __LINE__, __VA_ARGS__)
#define checkRuntime(call)                                                                 \
  do {                                                                                     \
    auto ___call__ret_code__ = (call);                                                     \
    if (___call__ret_code__ != cudaSuccess) {                                              \
      INFO("CUDA Runtime error💥 %s # %s, code = %s [ %d ]", #call,                         \
           cudaGetErrorString(___call__ret_code__), cudaGetErrorName(___call__ret_code__), \
           ___call__ret_code__);                                                           \
      abort();                                                                             \
    }                                                                                      \
  } while (0)

#define checkKernel(...)                 \
  do {                                   \
    { (__VA_ARGS__); }                   \
    checkRuntime(cudaPeekAtLastError()); \
  } while (0)


__global__
void cal_Lxy(const uint8_t *Sxy, float *Lxy, int row, int col, int gauss_width, const float *gauss_kernel) {
    unsigned int tx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int ty = threadIdx.y + blockDim.y * blockIdx.y;
    if (tx >= col || ty >= row) return;
    for (int ch = 0; ch < 3; ch++) {
        float color = 0;
        for (int i = 0; i < gauss_width; ++i) {
            for (int j = 0; j < gauss_width; ++j) {
                unsigned int x = min(max(tx - gauss_width / 2 + j, 0), col - 1);
                unsigned int y = min(max(ty - gauss_width / 2 + i, 0), row - 1);
                float tem = gauss_kernel[i * gauss_width + j];
                color += tem * Sxy[(x + y * col) * 3 + ch];
            }
        }
        Lxy[(tx + ty * col) * 3 + ch] = color;
    }
}

__global__
void cal_LgRxy(const uint8_t *Sxy, const float *Lxy, float *LgRxy, int row, int col) {
    unsigned int tx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int ty = threadIdx.y + blockIdx.y * blockDim.y;
    if (tx >= col || ty >= row) return;
    for (int ch = 0; ch < 3; ch++) {
        float l = Lxy[(tx + ty * col) * 3 + ch], s = Sxy[(tx + ty * col) * 3 + ch];
        if (l < 0.01) l = 0.01;
        if (s < 0.01) s = 0.01;
        float val = log10(s) - log10(l);
        LgRxy[(tx + ty * col) * 3 + ch] = val;
    }
}

__global__
void cudaGetMinMax(float *LgRxy, int row, int col, float *arrayMin, float *arrayMax) {
    unsigned int tx = threadIdx.x + blockDim.x * blockIdx.x;
    if (tx >= row) return;
    for (int ch = 0; ch < 3; ch++) {
        float ma = 0, mi = 0;
        for (int i = 0; i < col; ++i) {
            ma = max(LgRxy[(i + tx * col) * 3 + ch], ma);
            mi = min(LgRxy[(i + tx * col) * 3 + ch], mi);
        }
        arrayMax[tx * 3 + ch] = ma;
        arrayMin[tx * 3 + ch] = mi;
    }
}

__global__
void cal_Rxy(const float *LgRxy, uint8_t *Rxy, int row, int col, const float *r_min,
             const float *r_max) {
    unsigned int tx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int ty = threadIdx.y + blockIdx.y * blockDim.y;
    if (tx >= col || ty >= row) return;
    for (int ch = 0; ch < 3; ch++) {
        uint8_t val = (LgRxy[(tx + ty * col) * 3 + ch] - r_min[ch]) * 255 / (r_max[ch] - r_min[ch]);
        Rxy[(tx + ty * col) * 3 + ch] = val;
    }
}

void SSR(const cv::Mat &src, cv::Mat &dst, float sigma) {
    dst = cv::Mat(src.size(), CV_8UC3);
    int kSize = (int) (sigma * 3 / 2);
    kSize = kSize * 2 + 1;
    float *Lxy, *LgRxy, *gauss_kernel;
    uint8_t *Rxy, *Sxy;
    checkRuntime(cudaMallocManaged(&Sxy, sizeof(uint8_t) * src.rows * src.cols * 3));
    checkRuntime(cudaMallocManaged(&LgRxy, sizeof(float) * src.rows * src.cols * 3));
    checkRuntime(cudaMallocManaged(&Lxy, sizeof(float) * src.rows * src.cols * 3));
    checkRuntime(cudaMallocManaged(&Rxy, sizeof(uint8_t) * src.rows * src.cols * 3));
    checkRuntime(cudaMallocManaged(&gauss_kernel, sizeof(float) * kSize * kSize));


    cv::Mat ga = cv::getGaussianKernel(kSize * kSize, sigma, CV_32F);
    checkRuntime(cudaMemcpy(gauss_kernel, ga.data, sizeof(float) * kSize * kSize, cudaMemcpyHostToDevice));

    checkRuntime(cudaMemcpy(Sxy, src.data, src.rows * src.cols * 3, cudaMemcpyHostToDevice));
    checkKernel(
            cal_Lxy<<<dim3((src.cols + 31) / 32, (src.rows + 31) / 32), dim3(32, 32)>>>(Sxy, Lxy, src.rows, src.cols,
                                                                                        kSize,
                                                                                        gauss_kernel));
    checkKernel(cal_LgRxy<<<dim3((src.cols + 31) / 32, (src.rows + 31) / 32), dim3(32, 32)>>>(Sxy, Lxy, LgRxy, src.rows,
                                                                                              src.cols));
//    auto *arrayMin = new float[src.rows * 3];
//    auto *arrayMax = new float[src.rows * 3];
    float *arrayMin_cuda, *arrayMax_cuda;
    checkRuntime(cudaMallocManaged(&arrayMax_cuda, sizeof(float) * src.rows * 3));
    checkRuntime(cudaMallocManaged(&arrayMin_cuda, sizeof(float) * src.rows * 3));
    checkKernel(cudaGetMinMax<<<(src.rows + 31) / 32, 32>>>(LgRxy, src.rows, src.cols, arrayMin_cuda, arrayMax_cuda));
//    checkRuntime(cudaMemcpy(arrayMin, arrayMin_cuda, sizeof(float) * src.rows, cudaMemcpyDeviceToHost));
//    checkRuntime(cudaMemcpy(arrayMax, arrayMax_cuda, sizeof(float) * src.rows, cudaMemcpyDeviceToHost));
    float ma[3] = {0, 0, 0}, mi[3] = {0, 0, 0};
    for (int ch = 0; ch < 3; ch++) {
        for (int i = 0; i < src.rows; ++i) {
            ma[ch] = std::max(ma[ch], arrayMax_cuda[i * 3 + ch]);
            mi[ch] = std::min(mi[ch], arrayMin_cuda[i * 3 + ch]);
        }
    }
    float *ma_cuda, *mi_cuda;
    checkRuntime(cudaMalloc(&ma_cuda, sizeof(float) * 3));
    checkRuntime(cudaMalloc(&mi_cuda, sizeof(float) * 3));
    checkRuntime(cudaMemcpy(ma_cuda, ma, sizeof(float) * 3, cudaMemcpyHostToDevice));
    checkRuntime(cudaMemcpy(mi_cuda, mi, sizeof(float) * 3, cudaMemcpyHostToDevice));
    checkKernel(
            cal_Rxy<<<dim3((src.cols + 31) / 32, (src.rows + 31) / 32), dim3(32, 32)>>>(LgRxy, Rxy, src.rows, src.cols,
                                                                                        mi_cuda,
                                                                                        ma_cuda));
    checkRuntime(cudaMemcpy(dst.data, Rxy, sizeof(uint8_t) * dst.rows * dst.cols * 3, cudaMemcpyDeviceToHost));
    checkRuntime(cudaDeviceSynchronize());
    cudaFree(Sxy);
    cudaFree(LgRxy);
    cudaFree(Lxy);
    cudaFree(gauss_kernel);
    cudaFree(Rxy);
    cudaFree(arrayMin_cuda);
    cudaFree(arrayMax_cuda);
    cudaFree(ma_cuda);
    cudaFree(mi_cuda);
}

int main() {
    cv::Mat img = cv::imread("/home/liangfuchu/code/cpp/vision_cpp/images/img.png");
    cv::Mat dst;
    SSR(img, dst, 30);
    cv::imshow("output", dst);
    cv::waitKey(0);
}