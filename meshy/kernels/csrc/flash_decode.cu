// sm70 flash-decoding paged decode attention (no tensor cores).
//
// Grid: (seq * kv_head, num_splits). Block = 128 threads = 4 warps =
// 8 half-warps. Each half-warp (16 lanes) owns one key at a time; a
// lane loads 8 fp16 K/V components (one 16 B uint4) of one key, so a
// warp streams two keys per iteration. QK is reduced inside the
// 16-lane group with four shfl_xor(width=16) — no shared memory or
// block sync on the hot path. PV leaves 8 fp32 accumulators per lane
// per q head (16 total; GQA group size 2 shares the K/V read). Each
// half-warp keeps its own online-softmax state; the 8 sub-results are
// merged once at block end through shared memory, then a second kernel
// combines the KV splits by log-sum-exp.
//
// KV slots follow SGLang's flat decode metadata:
//   slot = kv_indices[kv_indptr[seq] + token]
// and k/v point at the c128 pool reshaped to [num_slots, kv_heads, D].

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>

constexpr int HEAD_DIM = 128;
constexpr int GROUP_SIZE = 2;       // q heads per kv head (Qwen3-0.6B)
constexpr int LANE_GROUP = 16;     // lanes per key
constexpr int DIMS_PER_LANE = HEAD_DIM / LANE_GROUP;  // 8
constexpr int HALF_WARPS = 8;      // 4 warps * 2 keys/warp
constexpr float NEG_INF = -3.0e38f;

__device__ __forceinline__ float shfl_group(float v, int lane) {
  // Width 16 confines the reduction to the half-warp owning one key.
  return __shfl_xor_sync(0xFFFFFFFF, v, lane, LANE_GROUP);
}

__device__ __forceinline__ void load_vec(const __half* p, float out[DIMS_PER_LANE]) {
  // One 16 B transaction: 8 fp16 values as four __half2 pairs.
  const __half2* h2 = reinterpret_cast<const __half2*>(p);
#pragma unroll
  for (int i = 0; i < DIMS_PER_LANE / 2; ++i) {
    __half2 a = h2[i];
    out[2 * i] = __low2float(a);
    out[2 * i + 1] = __high2float(a);
  }
}

// Partial kernel: one block per (seq*kv_head, split).
__global__ void flash_partial_kernel(
    const __half* __restrict__ q,
    const __half* __restrict__ k_flat,
    const __half* __restrict__ v_flat,
    const int* __restrict__ kv_indptr,
    const int* __restrict__ kv_indices,
    int num_kv_heads, float scale, int chunk_tokens,
    __half* __restrict__ partial_out, float* __restrict__ partial_lse) {
  const int block_id = blockIdx.x;
  const int seq = block_id / num_kv_heads;
  const int kvh = block_id % num_kv_heads;
  const int split = blockIdx.y;

  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int half = lane / LANE_GROUP;       // 0 or 1 within the warp
  const int glane = lane % LANE_GROUP;      // lane inside key group
  const int hw = warp * 2 + half;           // half-warp id 0..7
  const int base_dim = glane * DIMS_PER_LANE;

  // Row length comes from the CSR pointers themselves, so no separate
  // length tensor / host sync is needed (CUDA-graph safe).
  const int indptr = kv_indptr[seq];
  const int seq_len = kv_indptr[seq + 1] - indptr;

  // Per-lane Q for the group's two q heads.
  float q0[DIMS_PER_LANE], q1[DIMS_PER_LANE];
  const __half* qp0 = q + ((int64_t)seq * (num_kv_heads * GROUP_SIZE)
                           + kvh * GROUP_SIZE) * HEAD_DIM + base_dim;
  const __half* qp1 = qp0 + HEAD_DIM;
  load_vec(qp0, q0);
  load_vec(qp1, q1);

  const int64_t kv_stride = (int64_t)num_kv_heads * HEAD_DIM;

  // Each half-warp owns a contiguous eighth of the chunk (host pads
  // chunk_tokens to a multiple of HALF_WARPS).
  const int keys_per_hw = chunk_tokens / HALF_WARPS;
  const int t_begin = split * chunk_tokens + hw * keys_per_hw;
  const int t_end = t_begin + keys_per_hw;

  float acc0[DIMS_PER_LANE], acc1[DIMS_PER_LANE];
#pragma unroll
  for (int i = 0; i < DIMS_PER_LANE; ++i) { acc0[i] = 0.f; acc1[i] = 0.f; }
  float m0 = NEG_INF, m1 = NEG_INF, l0 = 0.f, l1 = 0.f;

  // Software-pipelined K prefetch (raw 16 B).
  uint4 k_next{};
  int t_next = t_begin;
  auto vslot_ptr = [&](int slot) -> const __half* {
    return reinterpret_cast<const __half*>(
        &v_flat[(int64_t)slot * kv_stride + (int64_t)kvh * HEAD_DIM
                + base_dim]);
  };
  auto load_slot = [&](int t) -> uint4 {
    int slot = (t < seq_len) ? kv_indices[indptr + t] : 0;
    return *reinterpret_cast<const uint4*>(
        &k_flat[(int64_t)slot * kv_stride + (int64_t)kvh * HEAD_DIM
                + base_dim]);
  };
  if (t_next < t_end) k_next = load_slot(t_next);

  for (int t = t_begin; t < t_end; ++t) {
    uint4 k_raw = k_next;
    if (t + 1 < t_end) k_next = load_slot(t + 1);  // outstanding during math
    float kk[DIMS_PER_LANE];
    {
      const __half2* h2 = reinterpret_cast<const __half2*>(&k_raw);
#pragma unroll
      for (int i = 0; i < DIMS_PER_LANE / 2; ++i) {
        __half2 a = h2[i];
        kk[2 * i] = __low2float(a);
        kk[2 * i + 1] = __high2float(a);
      }
    }
    float s0 = 0.f, s1 = 0.f;
#pragma unroll
    for (int i = 0; i < DIMS_PER_LANE; ++i) { s0 += q0[i] * kk[i]; s1 += q1[i] * kk[i]; }
#pragma unroll
    for (int off = 1; off < LANE_GROUP; off <<= 1) {
      s0 += shfl_group(s0, off);
      s1 += shfl_group(s1, off);
    }
    const bool valid = t < seq_len;
    s0 = valid ? s0 * scale : NEG_INF;
    s1 = valid ? s1 * scale : NEG_INF;

    const float nm0 = fmaxf(m0, s0), nm1 = fmaxf(m1, s1);
    const float c0 = expf(m0 - nm0), c1 = expf(m1 - nm1);
    const float p0 = valid ? expf(s0 - nm0) : 0.f;
    const float p1 = valid ? expf(s1 - nm1) : 0.f;

    int slot = (valid) ? kv_indices[indptr + t] : 0;
    float vv[DIMS_PER_LANE];
    load_vec(vslot_ptr(slot), vv);
#pragma unroll
    for (int i = 0; i < DIMS_PER_LANE; ++i) {
      acc0[i] = acc0[i] * c0 + p0 * vv[i];
      acc1[i] = acc1[i] * c1 + p1 * vv[i];
    }
    m0 = nm0; m1 = nm1;
    l0 = l0 * c0 + p0;
    l1 = l1 * c1 + p1;
  }

  // ---- merge the 8 half-warp sub-results once via shared memory ----
  __shared__ float sm_m[HALF_WARPS][GROUP_SIZE];
  __shared__ float sm_l[HALF_WARPS][GROUP_SIZE];
  __shared__ float sm_acc[HALF_WARPS][GROUP_SIZE][LANE_GROUP][DIMS_PER_LANE];

  sm_m[hw][0] = m0; sm_m[hw][1] = m1;
  sm_l[hw][0] = l0; sm_l[hw][1] = l1;
#pragma unroll
  for (int i = 0; i < DIMS_PER_LANE; ++i) {
    sm_acc[hw][0][glane][i] = acc0[i];
    sm_acc[hw][1][glane][i] = acc1[i];
  }
  __syncthreads();

  // 32 threads: two q heads x 16 lane groups; each merges 8 sub-parts.
  if (tid < GROUP_SIZE * LANE_GROUP) {
    const int g = tid / LANE_GROUP;
    const int lg = tid % LANE_GROUP;
    float gmax = NEG_INF;
#pragma unroll
    for (int u = 0; u < HALF_WARPS; ++u) gmax = fmaxf(gmax, sm_m[u][g]);
    float denom = 0.f;
#pragma unroll
    for (int u = 0; u < HALF_WARPS; ++u)
      denom += (sm_l[u][g] > 0.f)
                   ? sm_l[u][g] * expf(sm_m[u][g] - gmax)
                   : 0.f;
    float out[DIMS_PER_LANE];
#pragma unroll
    for (int i = 0; i < DIMS_PER_LANE; ++i) out[i] = 0.f;
#pragma unroll
    for (int u = 0; u < HALF_WARPS; ++u) {
      if (sm_l[u][g] > 0.f) {
        float w = expf(sm_m[u][g] - gmax);
#pragma unroll
        for (int i = 0; i < DIMS_PER_LANE; ++i)
          out[i] += sm_acc[u][g][lg][i] * w;
      }
    }
    const int dim0 = lg * DIMS_PER_LANE;
    const int64_t part_head =
        (int64_t)split * gridDim.x + block_id;   // seq*kv_head blocks
    __half* op = partial_out
        + ((part_head * GROUP_SIZE + g) * HEAD_DIM + dim0);
#pragma unroll
    for (int i = 0; i < DIMS_PER_LANE; ++i)
      op[i] = __float2half(out[i] / fmaxf(denom, 1e-30f));
    if (lg == 0)
      partial_lse[part_head * GROUP_SIZE + g] =
          (denom > 0.f) ? gmax + logf(fmaxf(denom, 1e-30f)) : NEG_INF;
  }
}

// Combine KV splits: one block per (seq, q head), one thread per dim.
__global__ void flash_combine_kernel(
    const __half* __restrict__ partial_out,
    const float* __restrict__ partial_lse,
    int num_splits, int batch, int num_kv_heads,
    __half* __restrict__ out) {
  const int hb = blockIdx.x;      // seq * num_q_heads + q head
  const int seq = hb / (num_kv_heads * GROUP_SIZE);
  const int qh = hb % (num_kv_heads * GROUP_SIZE);
  const int kvh = qh / GROUP_SIZE;
  const int g = qh % GROUP_SIZE;
  const int kv_block = seq * num_kv_heads + kvh;
  const int kv_blocks = batch * num_kv_heads;
  const int d = threadIdx.x;
  auto lse_at = [&](int sp) -> float {
    return partial_lse[((int64_t)sp * kv_blocks + kv_block) * GROUP_SIZE + g];
  };
  auto out_at = [&](int sp) -> const __half* {
    return partial_out
        + (((int64_t)sp * kv_blocks + kv_block) * GROUP_SIZE + g) * HEAD_DIM;
  };
  float gmax = NEG_INF;
  for (int sp = 0; sp < num_splits; ++sp) gmax = fmaxf(gmax, lse_at(sp));
  float denom = 0.f, acc = 0.f;
  for (int sp = 0; sp < num_splits; ++sp) {
    float lse = lse_at(sp);
    if (lse > NEG_INF) {
      float w = expf(lse - gmax);
      denom += w;
      acc += w * __half2float(out_at(sp)[d]);
    }
  }
  out[(int64_t)hb * HEAD_DIM + d] =
      __float2half(acc / fmaxf(denom, 1e-30f));
}

void launch_flash_decode(
    torch::Tensor q, torch::Tensor k_flat, torch::Tensor v_flat,
    torch::Tensor kv_indptr, torch::Tensor kv_indices,
    int64_t grid_batch, int64_t num_splits, int64_t chunk_tokens,
    double scale, torch::Tensor partial_out, torch::Tensor partial_lse,
    torch::Tensor out) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int num_q_heads = q.size(1);
  const int num_kv_heads = num_q_heads / GROUP_SIZE;
  const int head_blocks = (int)grid_batch * num_kv_heads;
  dim3 grid(head_blocks, (unsigned)num_splits);
  flash_partial_kernel<<<grid, 128, 0, stream>>>(
      reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(k_flat.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(v_flat.data_ptr<at::Half>()),
      kv_indptr.data_ptr<int>(), kv_indices.data_ptr<int>(),
      num_kv_heads, (float)scale, (int)chunk_tokens,
      reinterpret_cast<__half*>(partial_out.data_ptr<at::Half>()),
      partial_lse.data_ptr<float>());
  flash_combine_kernel<<<(int)grid_batch * num_q_heads, 128, 0, stream>>>(
      reinterpret_cast<const __half*>(partial_out.data_ptr<at::Half>()),
      partial_lse.data_ptr<float>(), (int)num_splits, (int)grid_batch,
      num_kv_heads,
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("launch_flash_decode", &launch_flash_decode);
}
