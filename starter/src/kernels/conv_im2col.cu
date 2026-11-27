// conv_im2col.cu - explicit im2col lowering followed by a naive GEMM with the fused epilogue (Tier 1 reference).
//   A[M][Kdim]  with M = B*OH*OW (one row per output pixel) and Kdim = k*k*C in (kh, kw, ci) order,
//   W[N][Kdim]  is the weight blob as stored ([co][kh][kw][ci]), so out[m][n] = sum_k A[m][k] * W[n][k].
// Both kernels are one-thread-per-element and read global memory only. The GEMM is MMULT v0 from L3;
// your SMEM-tiled / register-tiled GEMM drops in here first, and implicit GEMM (Tier 2) removes A altogether.
#include "kernels.h"

template <typename T>
__global__ void im2col_kernel(ConvParams p, const T* __restrict__ in, T* __restrict__ A) {
    long long Kdim = (long long)p.k * p.k * p.C;
    long long M = (long long)p.B * p.OH * p.OW;
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= M * Kdim) return;
    long long m = i / Kdim, kk = i % Kdim;
    int ci = (int)(kk % p.C);
    int kw = (int)((kk / p.C) % p.k);
    int kh = (int)(kk / ((long long)p.C * p.k));
    int ox = (int)(m % p.OW);
    int oy = (int)((m / p.OW) % p.OH);
    int b = (int)(m / ((long long)p.OW * p.OH));
    int iy = oy * p.s + kh - p.p, ix = ox * p.s + kw - p.p;
    bool valid = (iy >= 0 && iy < p.H && ix >= 0 && ix < p.W);
    A[i] = valid ? in[(((long long)b * p.H + iy) * p.W + ix) * p.C + ci] : from_f32<T>(0.0f);
}

template <typename T>
__global__ void gemm_naive_epilogue_kernel(long long M, int N, long long Kdim, const T* __restrict__ A, const T* __restrict__ W,
                                           const T* __restrict__ bias, const T* __restrict__ skip, T* __restrict__ out, Epi epi) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= M * N) return;
    int n = (int)(i % N);
    long long m = i / N;
    const T* a = A + m * Kdim;
    const T* w = W + (long long)n * Kdim;
    float acc = to_f32(bias[n]);
    for (long long kk = 0; kk < Kdim; ++kk) acc += to_f32(a[kk]) * to_f32(w[kk]);
    if (epi == Epi::AddRelu) acc += to_f32(skip[i]);
    if (epi != Epi::None) acc = fmaxf(acc, 0.0f);
    out[i] = from_f32<T>(acc);
}
// gemm tiled implemetation
template <typename T, int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM/TM) * (BN/TN)) gemm_tiled_epilogue_kernel(long long M, int N, long long Kdim, const T* __restrict__ A, const T* __restrict__ W,
                                           const T* __restrict__ bias, const T* __restrict__ skip, T* __restrict__ out, Epi epi) {
    constexpr int NTX = BN / TN, NTY = BM / TM, NT = NTX * NTY;
    constexpr int PAD = 4;                         // makes the transposed stores conflict-free
    __shared__ float As[BK][BM + PAD];             // k-major
    __shared__ float Bs[BK][BN + PAD];

    const int tid = threadIdx.x;
    const int tx = tid % NTX, ty = tid / NTX;
    const long long m0 = (long long)blockIdx.x * BM;   // M tiles on x: consecutive blocks share the B tile (L2 reuse)
    const int n0 = blockIdx.y * BN;

    float acc[TM][TN] = {};
    float a[TM], b[TN];

    for (long long k0 = 0; k0 < Kdim; k0 += BK) {
        // cooperative load: k is the fastest index, so each row segment is contiguous in GMEM
        #pragma unroll
        for (int idx = tid; idx < BM * BK; idx += NT) {
            int r = idx / BK, k = idx % BK;
            long long gm = m0 + r, gk = k0 + k;
            As[k][r] = (gm < M && gk < Kdim) ? to_f32(A[gm * Kdim + gk]) : 0.f;
        }
        #pragma unroll
        for (int idx = tid; idx < BN * BK; idx += NT) {
            int r = idx / BK, k = idx % BK;
            int gn = n0 + r; long long gk = k0 + k;
            Bs[k][r] = (gn < N && gk < Kdim) ? to_f32(W[(long long)gn * Kdim + gk]) : 0.f;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < TM / 2; ++i) {
                a[i]          = As[k][ty * (TM / 2) + i];
                a[i + TM / 2] = As[k][BM / 2 + ty * (TM / 2) + i];
            }
            #pragma unroll
            for (int j = 0; j < TN / 2; ++j) {
                b[j]          = Bs[k][tx * (TN / 2) + j];
                b[j + TN / 2] = Bs[k][BN / 2 + tx * (TN / 2) + j];
            }
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

    // fused epilogue, identical semantics to the naive kernel
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        long long m = m0 + (i < TM / 2 ? ty * (TM / 2) + i : BM / 2 + ty * (TM / 2) + i - TM / 2);
        if (m >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int n = n0 + (j < TN / 2 ? tx * (TN / 2) + j : BN / 2 + tx * (TN / 2) + j - TN / 2);
            if (n >= N) continue;
            float v = acc[i][j] + to_f32(bias[n]);
            if (epi == Epi::AddRelu) v += to_f32(skip[m * N + n]);
            if (epi != Epi::None) v = fmaxf(v, 0.f);
            out[m * N + n] = from_f32<T>(v);
        }
    }
}

template <typename T, int BM, int BN, int TM, int TN>
static void launch_tiled(long long M, int N, long long Kdim, const T* A, const T* w,
                         const T* bias, const T* skip, T* out, Epi epi) {
    dim3 grid((unsigned)((M + BM - 1) / BM), (N + BN - 1) / BN);
    dim3 block((BM / TM) * (BN / TN));
    gemm_tiled_epilogue_kernel<T, BM, BN, 8, TM, TN><<<grid, block>>>(M, N, Kdim, A, w, bias, skip, out, epi);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void launch_conv_im2col(const ConvParams& p, const T* in, const T* w, const T* bias, const T* skip, T* out, T* scratchA) {
    long long Kdim = (long long)p.k * p.k * p.C;
    long long M = (long long)p.B * p.OH * p.OW;
    LAUNCH(im2col_kernel<T>, grid1d(M * Kdim), dim3(kDefaultBlock), p, in, scratchA);
    CUDA_CHECK(cudaGetLastError());
    //LAUNCH(gemm_naive_epilogue_kernel<T>, grid1d(M * p.K), dim3(kDefaultBlock), M, p.K, Kdim, (const T*)scratchA, w, bias, skip, out, p.epi);
    if (p.K >= 128) {
        launch_tiled<T,128,128,8,8>(M, p.K, Kdim, scratchA, w, bias, skip, out, p.epi);
    }
    else {
        launch_tiled<T,64,64,8,8>(M, p.K, Kdim, scratchA, w, bias, skip, out, p.epi);
    }
    CUDA_CHECK(cudaGetLastError());
}

template void launch_conv_im2col<float>(const ConvParams&, const float*, const float*, const float*, const float*, float*, float*);
#ifdef ENABLE_FP16
template void launch_conv_im2col<half>(const ConvParams&, const half*, const half*, const half*, const half*, half*, half*);
#endif
