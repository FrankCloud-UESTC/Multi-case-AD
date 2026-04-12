/**
 * Fused distance heatmap CUDA kernel for anomaly detection.
 *
 * Contains two implementations:
 *   1. Naive: one thread per query, loops over all keys from global memory
 *   2. Tiled: shared memory tiling for keys - reduces global memory traffic
 *             by a factor of blockDim.x (each key tile loaded once per block,
 *             shared by all threads)
 *
 * Shared memory tiling is critical for larger N because:
 *   - Naive: each thread reads M*D floats from keys = N*M*D total global reads
 *   - Tiled: each key tile loaded once per block = (N/blockDim.x)*M*D total reads
 *   - For N=4096, blockDim=256: 16x reduction in global memory traffic for keys
 */

#include <torch/extension.h>
#include <cuda_runtime.h>

// ============================================================================
// Kernel 1: Naive (original) - one thread per query, global memory only
// ============================================================================

__global__ void compute_heatmap_naive_kernel(
    const float* __restrict__ query,
    const float* __restrict__ keys,
    float* __restrict__ output,
    int N, int M, int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const float* q = query + idx * D;

    float min_dist = 1e30f;
    int nearest_k = 0;

    for (int k = 0; k < M; k++) {
        const float* key = keys + k * D;
        float dist = 0.0f;
        for (int d = 0; d < D; d++) {
            float diff = q[d] - key[d];
            dist += diff * diff;
        }
        dist /= D;
        if (dist < min_dist) {
            min_dist = dist;
            nearest_k = k;
        }
    }

    const float* nearest_key = keys + nearest_k * D;
    float score = 0.0f;
    for (int d = 0; d < D; d++) {
        float diff = q[d] - nearest_key[d];
        float diff2 = diff * diff;
        score += diff2 * diff2;
    }

    output[idx] = score;
}

// ============================================================================
// Kernel 2: Shared memory tiled - loads key tiles into shared memory
// ============================================================================
// TILE_SIZE is the number of keys loaded per tile.
// Shared memory per block = TILE_SIZE * D * sizeof(float)
// For D=512, TILE_SIZE=16: 16 * 512 * 4 = 32KB (within 48KB limit)

template <int TILE_SIZE>
__global__ void compute_heatmap_tiled_kernel(
    const float* __restrict__ query,
    const float* __restrict__ keys,
    float* __restrict__ output,
    int N, int M, int D
) {
    // Dynamic shared memory: TILE_SIZE * D floats
    extern __shared__ float s_keys[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int bdim = blockDim.x;

    // Early exit for out-of-bounds threads (but still participate in sync)
    bool valid = (idx < N);

    const float* q = valid ? (query + idx * D) : nullptr;

    float min_dist = 1e30f;
    int nearest_k = 0;

    // Process keys in tiles of TILE_SIZE
    for (int tile_start = 0; tile_start < M; tile_start += TILE_SIZE) {
        int tile_end = tile_start + TILE_SIZE;
        if (tile_end > M) tile_end = M;
        int cur_tile = tile_end - tile_start;

        // Cooperatively load key tile into shared memory
        // Each thread loads elements strided by bdim
        int total_floats = cur_tile * D;
        for (int i = tid; i < total_floats; i += bdim) {
            int k = i / D;         // which key within the tile
            int d = i % D;         // which dimension
            s_keys[i] = keys[(tile_start + k) * D + d];
        }
        __syncthreads();

        // Compute distances to all keys in this tile
        if (valid) {
            for (int k = 0; k < cur_tile; k++) {
                const float* s_key = s_keys + k * D;
                float dist = 0.0f;
                for (int d = 0; d < D; d++) {
                    float diff = q[d] - s_key[d];
                    dist += diff * diff;
                }
                dist /= D;
                if (dist < min_dist) {
                    min_dist = dist;
                    nearest_k = tile_start + k;
                }
            }
        }
        __syncthreads();
    }

    // Compute anomaly score using the nearest key
    if (valid) {
        // Re-read nearest key from global memory (likely in L2 cache)
        const float* nearest_key = keys + nearest_k * D;
        float score = 0.0f;
        for (int d = 0; d < D; d++) {
            float diff = q[d] - nearest_key[d];
            float diff2 = diff * diff;
            score += diff2 * diff2;
        }
        output[idx] = score;
    }
}

// ============================================================================
// PyTorch bindings
// ============================================================================

torch::Tensor compute_heatmap_cuda(
    torch::Tensor query,   // (B, H, W, D)
    torch::Tensor keys     // (M, D)
) {
    auto query_2d = query.contiguous().view({-1, query.size(-1)});
    int N = query_2d.size(0);
    int D = query_2d.size(1);
    int M = keys.size(0);

    auto output = torch::zeros({N}, query.options());

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    compute_heatmap_naive_kernel<<<blocks, threads>>>(
        query_2d.data_ptr<float>(),
        keys.data_ptr<float>(),
        output.data_ptr<float>(),
        N, M, D
    );

    return output.view({query.size(0), query.size(1), query.size(2), 1});
}

torch::Tensor compute_heatmap_tiled_cuda(
    torch::Tensor query,   // (B, H, W, D)
    torch::Tensor keys     // (M, D)
) {
    auto query_2d = query.contiguous().view({-1, query.size(-1)});
    int N = query_2d.size(0);
    int D = query_2d.size(1);
    int M = keys.size(0);

    auto output = torch::zeros({N}, query.options());

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // TILE_SIZE=16 keys per tile, shared memory = 16 * D * 4 bytes
    const int TILE_SIZE = 16;
    int shared_mem_size = TILE_SIZE * D * sizeof(float);

    compute_heatmap_tiled_kernel<TILE_SIZE><<<blocks, threads, shared_mem_size>>>(
        query_2d.data_ptr<float>(),
        keys.data_ptr<float>(),
        output.data_ptr<float>(),
        N, M, D
    );

    return output.view({query.size(0), query.size(1), query.size(2), 1});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_heatmap", &compute_heatmap_cuda, "Fused anomaly heatmap - naive (CUDA)");
    m.def("compute_heatmap_tiled", &compute_heatmap_tiled_cuda, "Fused anomaly heatmap - shared memory tiled (CUDA)");
}
