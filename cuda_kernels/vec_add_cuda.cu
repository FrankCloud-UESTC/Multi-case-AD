#include <torch/extension.h>

// 简单的向量加法 CUDA kernel
__global__ void vec_add_kernel(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

// PyTorch 绑定函数
torch::Tensor vec_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto out = torch::zeros_like(a);
    int n = a.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vec_add_kernel<<<blocks, threads>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vec_add", &vec_add_cuda, "Vector addition (CUDA)");
}
