#include "msm/msm.cuh"

#include <cooperative_groups.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda.h>

#include <iostream>
#include <stdexcept>
#include <vector>

#include <thrust/sort.h>  // 添加thrust头文件
#include <thrust/execution_policy.h>  // 添加执行策略头文件
#include "curves/affine.cuh"
#include "curves/projective.cuh"
#include "fields/field.cuh"
#include "gpu-utils/error_handler.cuh"
#include "utils/mont.cuh"

namespace msm {

  namespace {

#define MAX_TH 256

    // #define SSM_SUM  //WIP

    template <typename A, typename P>
    __global__ void left_shift_kernel(A* points, const unsigned shift, const unsigned count, A* points_out)
    {
      const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
      if (tid >= count) return;
      P point = P::from_affine(points[tid]);
      for (unsigned i = 0; i < shift; i++)
        point = P::dbl(point);
      points_out[tid] = P::to_affine(point);
    }

    unsigned get_optimal_c(int bitsize) { return (unsigned)max(ceil(std::log2(bitsize)) - 4.0, 1.0); }

    template <typename E>
    __global__ void normalize_kernel(E* inout, E factor, int n)
    {
      int tid = blockIdx.x * blockDim.x + threadIdx.x;
      if (tid < n) inout[tid] = (inout[tid] + factor - 1) / factor;
    }

    // a kernel that writes to bucket_indices which enables large bucket accumulation to happen afterwards.
    // specifically we map thread indices to buckets which said threads will handle in accumulation.
    template <typename P>
    __global__ void initialize_large_bucket_indices(
      unsigned* sorted_bucket_sizes_sum,
      unsigned nof_pts_per_thread,
      unsigned nof_large_buckets,
      // log_nof_buckets_to_compute should be equal to ceil(log(nof_buckets_to_compute))
      unsigned log_nof_large_buckets,
      unsigned* bucket_indices)
    {
      const int tid = blockIdx.x * blockDim.x + threadIdx.x;
      if (tid >= nof_large_buckets) { return; }
      unsigned start = (sorted_bucket_sizes_sum[tid] + nof_pts_per_thread - 1) / nof_pts_per_thread + tid;
      unsigned end = (sorted_bucket_sizes_sum[tid + 1] + nof_pts_per_thread - 1) / nof_pts_per_thread + tid + 1;
      for (unsigned i = start; i < end; i++) {
        // this just concatenates two pieces of data - large bucket index and (i - start)
        bucket_indices[i] = tid | ((i - start) << log_nof_large_buckets);
      }
    }

    // this function provides a single step of reduction across buckets sizes of
    // which are given by large_bucket_sizes pointer
    template <typename P>
    __global__ void sum_reduction_variable_size_kernel(
      P* v,
      unsigned* bucket_sizes_sum,
      unsigned* bucket_sizes,
      unsigned* large_bucket_thread_indices,
      unsigned nof_threads)
    {
      const int tid = blockIdx.x * blockDim.x + threadIdx.x;
      if (tid >= nof_threads) { return; }

      unsigned large_bucket_tid = large_bucket_thread_indices[tid];
      unsigned segment_ind = tid - bucket_sizes_sum[large_bucket_tid] - large_bucket_tid;
      unsigned large_bucket_size = bucket_sizes[large_bucket_tid];
      if (segment_ind < (large_bucket_size >> 1)) { v[tid] = v[tid] + v[tid + ((large_bucket_size + 1) >> 1)]; }
    }

    template <typename P>
    __global__ void single_stage_multi_reduction_kernel(
      const P* v,
      P* v_r,
      unsigned orig_block_size,
      unsigned block_size,
      unsigned write_stride,
      unsigned buckets_per_bm,
      unsigned write_phase,
      unsigned step,
      unsigned nof_threads)
    {
      const int tid = blockIdx.x * blockDim.x + threadIdx.x;
      if (tid >= nof_threads) return; // 如果线程ID超出范围，则直接返回

      // 我们需要偏移的tid，因为我们不想减少到零桶，这允许跳过它们。
      // 对于write_phase==1，读取模式不同，因此我们不会跳过任何内容。
      const int shifted_tid = write_phase ? tid : tid + (tid + step) / step;
      const int jump = block_size / 2; // 每个块的跳跃大小为块大小的一半
      const int block_id = shifted_tid / jump; // 计算当前线程所属的块ID
      // 这里偏移的原因与shifted_tid相同，但我们跳过整个块，这仅在write_phase=1时发生，因为其读取模式。
      const int shifted_block_id = write_phase ? block_id + (block_id + step) / step : block_id;
      const int block_tid = shifted_tid % jump; // 计算线程在块内的ID
      const unsigned read_ind = orig_block_size * shifted_block_id + block_tid; // 计算读取索引
      const unsigned write_ind = jump * shifted_block_id + block_tid; // 计算写入索引
      const unsigned v_r_key =
        write_stride ? ((write_ind / buckets_per_bm) * 2 + write_phase) * write_stride + write_ind % buckets_per_bm
                     : read_ind; // 计算结果数组的索引
      v_r[v_r_key] = v[read_ind] + v[read_ind + jump]; // 执行归约操作，将结果写入v_r
    }

    // this kernel performs single scalar multiplication
    // each thread multiplies a single scalar and point
    template <typename P, typename S>
    __global__ void ssm_kernel(const S* scalars, const P* points, P* results, unsigned N)
    {
      unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
      if (tid < N) results[tid] = scalars[tid] * points[tid];
    }

    // this kernel sums all the elements in a given vector using multiple threads
    template <typename P>
    __global__ void sum_reduction_kernel(P* v, P* v_r)
    {
      unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

      // Start at 1/2 block stride and divide by two each iteration
      for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        // Each thread does work unless it is further than the stride
        if (threadIdx.x < s) { v[tid] = v[tid] + v[tid + s]; }
        __syncthreads();
      }

      // Let the thread 0 for this block write the final result
      if (threadIdx.x == 0) { v_r[blockIdx.x] = v[tid]; }
    }

    // this kernel initializes the buckets with zero points
    // each thread initializes a different bucket
    template <typename P>
    __global__ void initialize_buckets_kernel(P* buckets, unsigned N)
    {
      unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
      if (tid < N) buckets[tid] = P::zero(); // zero point
    }

    // this kernel splits the scalars into digits of size c
    // each thread splits a single scalar into nof_bms digits
    template <typename S>
    __global__ void split_scalars_kernel(
      unsigned* buckets_indices,
      unsigned* point_indices,
      const S* scalars,
      unsigned nof_scalars,
      unsigned points_size,
      unsigned msm_size,
      unsigned nof_bms,
      unsigned bm_bitsize,
      unsigned c,
      unsigned precomputed_bms_stride)
    {
      unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
      if (tid >= nof_scalars) return;

      unsigned bucket_index;
      unsigned current_index;
      unsigned msm_index = tid / msm_size;
      const S& scalar = scalars[tid];
      for (unsigned bm = 0; bm < nof_bms; bm++) {
        const unsigned precomputed_index = bm / precomputed_bms_stride;
        const unsigned target_bm = bm % precomputed_bms_stride;

        bucket_index = scalar.get_scalar_digit(bm, c);
        current_index = bm * nof_scalars + tid;

        if (bucket_index != 0) {
          buckets_indices[current_index] =
            (msm_index << (c + bm_bitsize)) | (target_bm << c) |
            bucket_index; // the bucket module number and the msm number are appended at the msbs
        } else {
          buckets_indices[current_index] = 0; // will be skipped
        }
        point_indices[current_index] =
          tid % points_size + points_size * precomputed_index; // the point index is saved for later
      }
    }

    template <typename S>
    __global__ void
    find_cutoff_kernel(unsigned* v, unsigned size, unsigned cutoff, unsigned run_length, unsigned* result)
    {
      unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
      const unsigned nof_threads = (size + run_length - 1) / run_length;
      if (tid >= nof_threads) { return; }
      const unsigned start_index = tid * run_length;
      for (int i = start_index; i < min(start_index + run_length, size - 1); i++) {
        if (v[i] > cutoff && v[i + 1] <= cutoff) {
          result[0] = i + 1;
          return;
        }
      }
      if (tid == 0 && v[size - 1] > cutoff) { result[0] = size; }
    }

    // this kernel adds up the points in each bucket
    template <typename P, typename A>
    __global__ void accumulate_buckets_kernel(
      P* __restrict__ buckets,
      unsigned* __restrict__ bucket_offsets,
      unsigned* __restrict__ bucket_sizes,
      unsigned* __restrict__ single_bucket_indices,
      const unsigned* __restrict__ point_indices,
      A* __restrict__ points,
      const unsigned nof_buckets,
      const unsigned nof_buckets_to_compute,
      const unsigned msm_idx_shift,
      const unsigned c)
    {
      unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
      if (tid >= nof_buckets_to_compute) return;
      unsigned msm_index = single_bucket_indices[tid] >> msm_idx_shift;
      const unsigned single_bucket_index = (single_bucket_indices[tid] & ((1 << msm_idx_shift) - 1));
      unsigned bucket_index = msm_index * nof_buckets + single_bucket_index;
      const unsigned bucket_offset = bucket_offsets[tid];
      const unsigned bucket_size = bucket_sizes[tid];

      P bucket; // get rid of init buckets? no.. because what about buckets with no points
      for (unsigned i = 0; i < bucket_size;
           i++) { // add the relevant points starting from the relevant offset up to the bucket size
        unsigned point_ind = point_indices[bucket_offset + i];
        A point = points[point_ind];
        bucket =
          i ? (point == A::zero() ? bucket : bucket + point) : (point == A::zero() ? P::zero() : P::from_affine(point));
      }
      buckets[bucket_index] = bucket;
    }

    template <typename P, typename A>
    __global__ void accumulate_large_buckets_kernel(
      P* __restrict__ buckets,                // 输出：目标桶的指针，用于存储累加结果
      unsigned* __restrict__ bucket_offsets,   // 输入：每个桶的偏移量数组
      unsigned* __restrict__ bucket_sizes,     // 输入：每个桶的大小数组
      unsigned* __restrict__ large_bucket_thread_indices, // 输入：大桶的线程索引
      unsigned* __restrict__ point_indices,     // 输入：点的索引数组
      A* __restrict__ points,                   // 输入：点的数组
      const unsigned nof_buckets_to_compute,    // 输入：需要计算的桶的数量
      const unsigned c,                         // 输入：每个桶的位数
      const int points_per_thread,              // 输入：每个线程处理的点的数量
      const unsigned log_nof_buckets_to_compute, // 输入：计算桶数量的对数
      const unsigned nof_threads)               // 输入：总线程数量
    {
      unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x; // 计算当前线程的全局ID
      if (tid >= nof_threads) return; // 如果线程ID超出范围，直接返回

      // 计算当前桶的索引
      int bucket_segment_index = large_bucket_thread_indices[tid] >> log_nof_buckets_to_compute; // 获取桶段索引
      large_bucket_thread_indices[tid] &= ((1 << log_nof_buckets_to_compute) - 1); // 获取桶的索引
      int bucket_ind = large_bucket_thread_indices[tid]; // 当前桶的索引

      // 计算当前桶的偏移量和大小
      const unsigned bucket_offset = bucket_offsets[bucket_ind] + bucket_segment_index * points_per_thread; // 当前桶的偏移量
      const unsigned bucket_size = max(0, (int)bucket_sizes[bucket_ind] - bucket_segment_index * points_per_thread); // 当前桶的大小
      P bucket; // 用于存储当前桶的累加结果
      unsigned run_length = min(bucket_size, points_per_thread); // 计算当前线程要处理的点的数量

      // 累加点
      for (unsigned i = 0; i < run_length; i++) { // 从偏移量开始累加点
        unsigned point_ind = point_indices[bucket_offset + i]; // 获取点的索引
        A point = points[point_ind]; // 获取点的值
        // 累加点到桶中
        bucket = i ? (point == A::zero() ? bucket : bucket + point) : (point == A::zero() ? P::zero() : P::from_affine(point));
      }
      buckets[tid] = run_length ? bucket : P::zero(); // 将结果存储到目标桶中
    }

    template <typename P>
    __global__ void distribute_large_buckets_kernel(
      const P* large_buckets,                // 输入：大桶的指针
      P* buckets,                            // 输出：目标桶的指针
      const unsigned* sorted_bucket_sizes_sum, // 输入：已排序的桶大小的累积和
      const unsigned* single_bucket_indices, // 输入：单个桶的索引
      const unsigned size,                   // 输入：桶的数量
      const unsigned nof_buckets,            // 输入：桶的总数
      const unsigned msm_idx_shift)          // 输入：用于计算桶索引的位移
    {
      unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x; // 计算当前线程的全局ID
      if (tid >= size) { return; } // 如果线程ID超出范围，直接返回

      // 计算当前桶的索引
      unsigned msm_index = single_bucket_indices[tid] >> msm_idx_shift; // 通过位移获取MSM索引
      unsigned bucket_index = msm_index * nof_buckets + (single_bucket_indices[tid] & ((1 << msm_idx_shift) - 1)); // 计算桶索引
      unsigned large_bucket_index = sorted_bucket_sizes_sum[tid] + tid; // 计算大桶的索引
      buckets[bucket_index] = large_buckets[large_bucket_index]; // 将大桶的值分配到目标桶中
    }

    // 大三角求和内核：对每个桶模块进行求和
    // 每个线程处理一个桶模块（bucket module）
    template <typename P>
    __global__ void big_triangle_sum_kernel(const P* buckets, P* final_sums, unsigned nof_bms, unsigned c)
    {
      unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x; // 计算全局线程ID
      if (tid >= nof_bms) return; // 越界检查
      
      unsigned buckets_in_bm = (1 << c); // 每个桶模块包含 2^c 个桶
      P line_sum = buckets[(tid + 1) * buckets_in_bm - 1]; // 从最后一个桶开始初始化累加和
      final_sums[tid] = line_sum; // 存储初始值
      
      // 反向遍历桶模块中的桶（跳过最后一个已处理的桶）
      for (unsigned i = buckets_in_bm - 2; i > 0; i--) {
        line_sum = line_sum + buckets[tid * buckets_in_bm + i]; // 使用运行总和法累加
        final_sums[tid] = final_sums[tid] + line_sum; // 累加到最终结果
      }
    }

  template <typename P>
__global__ void optimized_big_triangle_sum_kernel(const P* buckets, P* final_sums, unsigned nof_bms, unsigned c)
{
    extern __shared__ P sdata[]; // 声明动态共享内存
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nof_bms) return;

    const unsigned buckets_in_bm = (1 << c);
    const unsigned offset = tid * buckets_in_bm;

    // 将桶数据预加载到共享内存
    for (unsigned i = threadIdx.x; i < buckets_in_bm; i += blockDim.x) {
        sdata[i] = buckets[offset + i];
    }
    __syncthreads();

    // 反向累加优化
    P line_sum = sdata[buckets_in_bm - 1];
    P final_sum = line_sum;

    #pragma unroll
    for (unsigned i = buckets_in_bm - 2; i > 0; i --) {
        line_sum = line_sum + sdata[i];
        final_sum = final_sum + line_sum;
    }

    final_sums[tid] = final_sum;
}

  template <typename P> 
  __global__ void optimized_big_triangle_sum_kernel2(
      const P* __restrict__ buckets, 
      P* __restrict__ final_sums, 
      unsigned nof_bms, 
      unsigned c
  ) {
    extern __shared__ P shared_sums[];
    unsigned bm_id = blockIdx.x;
    if (bm_id >= nof_bms) return;

    unsigned buckets_in_bm = 1 << c;
    unsigned bm_offset = bm_id * buckets_in_bm;

    unsigned tid = threadIdx.x;
    unsigned num_threads = blockDim.x;
    unsigned k_per_thread = (buckets_in_bm - 1 + num_threads - 1) / num_threads;
    unsigned start_k = 1 + tid * k_per_thread;
    unsigned end_k = min(start_k + k_per_thread, buckets_in_bm);

    P local_sum = P::zero();
    for (unsigned k = start_k; k < end_k; ++k) {
      local_sum = local_sum + buckets[bm_offset + k]; // 修正：直接累加点，无需乘法
    }

    shared_sums[tid] = local_sum;
    __syncthreads();

    // 分层归约（同前）
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
      if (tid < s) shared_sums[tid] = shared_sums[tid] + shared_sums[tid + s];
      __syncthreads();
    }

    if (tid == 0) final_sums[bm_id] = shared_sums[0];
  }
    // 标量乘法内核：将每个桶乘以其索引对应的标量
    // 每个线程处理一个桶
    template <typename P, typename S>
    __global__ void ssm_buckets_kernel(P* buckets, unsigned* single_bucket_indices, unsigned nof_buckets, unsigned c)
    {
      unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x; // 计算全局线程ID
      if (tid >= nof_buckets) return; // 越界检查
      
      unsigned bucket_index = single_bucket_indices[tid]; // 获取当前桶的全局索引
      S scalar_bucket_multiplier;
      // 构造标量乘数：取桶索引的低c位（去除桶模块索引部分）
      scalar_bucket_multiplier = { 
        bucket_index & ((1 << c) - 1), 0, 0, 0, 0, 0, 0, 0 
      };
      buckets[bucket_index] = scalar_bucket_multiplier * buckets[bucket_index]; // 执行标量乘法
    }

    // 最后归约内核：将中间结果归约为最终窗口和
    template <typename P>
    __global__ void last_pass_kernel(
      const P* final_buckets,  // 输入：最终桶数据
      P* final_sums,           // 输出：最终窗口和
      unsigned nof_sums_per_batch,  // 每个批次的窗口数
      unsigned batch_size,     // 批次大小
      unsigned nof_bms_per_batch, // 每个批次的桶模块数
      unsigned orig_c)         // 原始c值（每个窗口的位数）
    {
      unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x; // 全局线程ID
      if (tid >= nof_sums_per_batch * batch_size) return; // 越界检查
      
      // 计算批次内索引
      unsigned batch_index = tid / nof_sums_per_batch;
      unsigned batch_tid = tid % nof_sums_per_batch;
      
      // 计算桶模块索引和窗口内索引
      unsigned bm_index = batch_tid / orig_c;
      unsigned bm_tid = batch_tid % orig_c;
      
      // 通过位操作调整索引结构
      for (unsigned c = orig_c; c > 1;) {
        c = (c + 1) >> 1;  // 向上取整的二分法
        bm_index <<= 1;     // 左移扩大索引范围
        if (bm_tid >= c) {  // 处理高位部分
          bm_index++;
          bm_tid -= c;
        }
      }
      // 写入最终结果（选择第二个元素，可能因存储格式需要）
      final_sums[tid] = final_buckets[2 * (batch_index * nof_bms_per_batch + bm_index) + 1];
    }

    // 最终累加内核：使用双倍加法算法聚合最终结果
    // 每个线程处理一个MSM（多标量乘法）结果
    template <typename P, typename S>
    __global__ void final_accumulation_kernel(
      const P* final_sums,   // 输入：各窗口的最终和
      P* final_results,       // 输出：最终结果
      unsigned nof_msms,     // MSM总数
      unsigned nof_results,  // 每个MSM的中间结果数
      unsigned c)            // 窗口位数
    {
      unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x; // 全局线程ID
      if (tid >= nof_msms) return; // 越界检查
      
      P final_result = P::zero(); // 初始化零元素
      
      // 反向遍历中间结果（跳过已知的空窗口）
      for (unsigned i = nof_results; i > 1; i--) {
        // 加法步骤：累加窗口和
        final_result = final_result + final_sums[i - 1 + tid * nof_results];
        
        // 双倍步骤：执行c次点加倍操作
        for (unsigned j = 0; j < c; j++) {
          final_result = final_result + final_result; // 点加倍
        }
      }
      // 添加最后一个未处理的元素
      final_results[tid] = final_result + final_sums[tid * nof_results];
    }

    // this function computes msm using the bucket method
    template <typename S, typename P, typename A>
    cudaError_t bucket_method_msm(
      unsigned bitsize,
      unsigned c,
      const S* scalars,
      const A* points,
      unsigned batch_size,      // number of MSMs to compute
      unsigned single_msm_size, // number of elements per MSM (a.k.a N)
      unsigned nof_points,      // number of EC points in 'points' array. Must be either (1) single_msm_size if MSMs are
                                // sharing points or (2) single_msm_size*batch_size otherwise
      P* final_result,
      bool are_scalars_on_device,
      bool are_scalars_montgomery_form,
      bool are_points_on_device,
      bool are_points_montgomery_form,
      bool are_results_on_device,
      bool is_big_triangle,
      int large_bucket_factor,
      int precompute_factor,
      bool is_async,
      cudaStream_t stream)
    {
       CHK_INIT_IF_RETURN();

      const unsigned nof_scalars = batch_size * single_msm_size; // 计算标量的总数量，假设批处理之间不共享标量
      const bool is_nof_points_valid = ((single_msm_size * batch_size) % nof_points == 0); // 检查点的数量是否可以被单个 MSM 大小和批处理大小整除
      if (!is_nof_points_valid) {
        // 如果点的数量不合法，抛出参数无效的错误
        THROW_ICICLE_ERR(
          IcicleError_t::InvalidArgument, "bucket_method_msm: #points must be divisible by single_msm_size*batch_size");
      }

      const S* d_scalars; // 指向设备上标量的指针
      S* d_allocated_scalars = nullptr; // 指向分配的设备内标量内存的指针，初始为nullptr
      if (!are_scalars_on_device) { // 如果标量不在设备上
        // 将标量复制到 GPU
        CHK_IF_RETURN(cudaMallocAsync(&d_allocated_scalars, sizeof(S) * nof_scalars, stream)); // 在设备上分配内存
        CHK_IF_RETURN(
          cudaMemcpyAsync(d_allocated_scalars, scalars, sizeof(S) * nof_scalars, cudaMemcpyHostToDevice, stream)); // 异步复制标量到设备

        if (are_scalars_montgomery_form) { // 如果标量是蒙哥马利形式
          // 将标量从蒙哥马利形式转换
          CHK_IF_RETURN(mont::from_montgomery(d_allocated_scalars, nof_scalars, stream, d_allocated_scalars));
        }
        d_scalars = d_allocated_scalars; // 设置设备上标量的指针
      } else { // 如果标量已经在设备上
        if (are_scalars_montgomery_form) { // 如果标量是蒙哥马利形式
          // 在设备上分配内存并转换标量
          CHK_IF_RETURN(cudaMallocAsync(&d_allocated_scalars, sizeof(S) * nof_scalars, stream));
          CHK_IF_RETURN(mont::from_montgomery(scalars, nof_scalars, stream, d_allocated_scalars));
          d_scalars = d_allocated_scalars; // 设置设备上标量的指针
        } else {
          d_scalars = scalars; // 直接使用已经在设备上的标量
        }
      }

      unsigned total_bms_per_msm = (bitsize + c - 1) / c; // 计算每个 MSM 的总桶模块数，向上取整
      unsigned nof_bms_per_msm = (total_bms_per_msm - 1) / precompute_factor + 1; // 计算每个 MSM 的桶模块数量，考虑预计算因子
      unsigned input_indexes_count = nof_scalars * total_bms_per_msm; // 计算输入索引的总数量

      unsigned bm_bitsize = (unsigned)ceil(std::log2(nof_bms_per_msm)); // 计算桶模块的位大小，取对数并向上取整

      unsigned* bucket_indices; // 指向桶索引的设备指针
      unsigned* point_indices; // 指向点索引的设备指针
      unsigned* sorted_bucket_indices; // 指向排序后的桶索引的设备指针
      unsigned* sorted_point_indices; // 指向排序后的点索引的设备指针
      // 在设备上分配内存用于桶索引和点索引
      CHK_IF_RETURN(cudaMallocAsync(&bucket_indices, sizeof(unsigned) * input_indexes_count, stream));
      CHK_IF_RETURN(cudaMallocAsync(&point_indices, sizeof(unsigned) * input_indexes_count, stream));
      CHK_IF_RETURN(cudaMallocAsync(&sorted_bucket_indices, sizeof(unsigned) * input_indexes_count, stream));
      CHK_IF_RETURN(cudaMallocAsync(&sorted_point_indices, sizeof(unsigned) * input_indexes_count, stream));

      // 将标量拆分为数字
      unsigned NUM_THREADS = 1 << 10; // 设置线程数为1024
      unsigned NUM_BLOCKS = (nof_scalars + NUM_THREADS - 1) / NUM_THREADS; // 计算块数，确保所有标量都被处理

      // 启动拆分标量的CUDA内核，将标量拆分成桶索引和点索引
      split_scalars_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
        bucket_indices, point_indices, d_scalars, nof_scalars, nof_points, single_msm_size, total_bms_per_msm,
        bm_bitsize, c, nof_bms_per_msm);
      
      nof_points *= precompute_factor; // 更新点的数量，考虑预计算因子

      // ------------------------------ 处理标量的排序步骤开始 ----------------------------------
      // 排序索引 - 将索引从小到大排序，以便将属于每个桶的点分组在一起
      unsigned* sort_indices_temp_storage{}; // 临时存储空间用于排序
      size_t sort_indices_temp_storage_bytes; // 临时存储空间的字节大小
      // 倒数第二个参数是显式提供的默认值，以允许传递流
      // 详细信息请参阅：https://nvlabs.github.io/cub/structcub_1_1_device_radix_sort.html#a65e82152de448c6373ed9563aaf8af7e
      CHK_IF_RETURN(cub::DeviceRadixSort::SortPairs(
        sort_indices_temp_storage, sort_indices_temp_storage_bytes, bucket_indices, sorted_bucket_indices,
        point_indices, sorted_point_indices, input_indexes_count, 0, sizeof(unsigned) * 8, stream));
      
      // 在设备上分配临时存储空间
      CHK_IF_RETURN(cudaMallocAsync(&sort_indices_temp_storage, sort_indices_temp_storage_bytes, stream));
      
      // 再次调用 SortPairs 内核进行实际的排序操作
      CHK_IF_RETURN(cub::DeviceRadixSort::SortPairs(
        sort_indices_temp_storage, sort_indices_temp_storage_bytes, bucket_indices, sorted_bucket_indices,
        point_indices, sorted_point_indices, input_indexes_count, 0, sizeof(unsigned) * 8, stream));
      
      // 释放临时存储空间和未排序的索引
      CHK_IF_RETURN(cudaFreeAsync(sort_indices_temp_storage, stream));
      CHK_IF_RETURN(cudaFreeAsync(bucket_indices, stream));
      CHK_IF_RETURN(cudaFreeAsync(point_indices, stream));

      // 计算桶模块的数量和每个模块中的桶数量
      unsigned nof_bms_in_batch = nof_bms_per_msm * batch_size; // 计算批次中的桶模块数量
      // 减去 nof_bms_per_msm，因为每个桶模块中不包括零桶
      const unsigned nof_buckets = (nof_bms_per_msm << c) - nof_bms_per_msm; // 计算每个桶模块中的桶数量
      const unsigned total_nof_buckets = nof_buckets * batch_size; // 计算总桶数量

      // 查找每个桶的大小
      unsigned* single_bucket_indices; // 单个桶的索引
      unsigned* bucket_sizes; // 每个桶的大小
      unsigned* nof_buckets_to_compute; // 需要计算的桶数量
      // 这里及其他地方加1，因为仍然有零索引对应于零桶
      CHK_IF_RETURN(cudaMallocAsync(&single_bucket_indices, sizeof(unsigned) * (total_nof_buckets + 1), stream));
      CHK_IF_RETURN(cudaMallocAsync(&bucket_sizes, sizeof(unsigned) * (total_nof_buckets + 1), stream));
      CHK_IF_RETURN(cudaMallocAsync(&nof_buckets_to_compute, sizeof(unsigned), stream));
      
      unsigned* encode_temp_storage{}; // 临时存储空间用于编码
      size_t encode_temp_storage_bytes = 0; // 临时存储空间的字节大小
      
      // 运行长度编码，将排序后的桶索引编码为单个桶索引和桶大小
      CHK_IF_RETURN(cub::DeviceRunLengthEncode::Encode(
        encode_temp_storage, encode_temp_storage_bytes, sorted_bucket_indices, single_bucket_indices, bucket_sizes,
        nof_buckets_to_compute, input_indexes_count, stream));
      
      // 在设备上分配临时存储空间
      CHK_IF_RETURN(cudaMallocAsync(&encode_temp_storage, encode_temp_storage_bytes, stream));
      
      // 再次调用 Encode 内核进行实际的编码操作
      CHK_IF_RETURN(cub::DeviceRunLengthEncode::Encode(
        encode_temp_storage, encode_temp_storage_bytes, sorted_bucket_indices, single_bucket_indices, bucket_sizes,
        nof_buckets_to_compute, input_indexes_count, stream));
      
      // 释放临时存储空间和排序后的桶索引
      CHK_IF_RETURN(cudaFreeAsync(encode_temp_storage, stream));
      CHK_IF_RETURN(cudaFreeAsync(sorted_bucket_indices, stream));

      // get offsets - where does each new bucket begin
      unsigned* bucket_offsets;
      CHK_IF_RETURN(cudaMallocAsync(&bucket_offsets, sizeof(unsigned) * (total_nof_buckets + 1), stream));
      unsigned* offsets_temp_storage{};
      size_t offsets_temp_storage_bytes = 0;
      CHK_IF_RETURN(cub::DeviceScan::ExclusiveSum(
        offsets_temp_storage, offsets_temp_storage_bytes, bucket_sizes, bucket_offsets, total_nof_buckets + 1, stream));
      CHK_IF_RETURN(cudaMallocAsync(&offsets_temp_storage, offsets_temp_storage_bytes, stream));
      CHK_IF_RETURN(cub::DeviceScan::ExclusiveSum(
        offsets_temp_storage, offsets_temp_storage_bytes, bucket_sizes, bucket_offsets, total_nof_buckets + 1, stream));
      CHK_IF_RETURN(cudaFreeAsync(offsets_temp_storage, stream));

      // ----------- 开始上传点（如果它们在主机上）并行于标量排序 ----------------
      const A* d_points; // 指向设备上点的指针
      A* d_allocated_points = nullptr; // 指向分配的设备内点内存的指针，初始为nullptr
      cudaStream_t stream_points = nullptr; // 点上传使用的CUDA流
      if (!are_points_on_device || are_points_montgomery_form) CHK_IF_RETURN(cudaStreamCreate(&stream_points)); // 如果点不在设备上或是蒙哥马利形式，创建新的CUDA流
      if (!are_points_on_device) { // 如果点不在设备上
        // 将点复制到GPU
        CHK_IF_RETURN(cudaMallocAsync(&d_allocated_points, sizeof(A) * nof_points, stream_points)); // 在设备上分配内存
        CHK_IF_RETURN(
          cudaMemcpyAsync(d_allocated_points, points, sizeof(A) * nof_points, cudaMemcpyHostToDevice, stream_points)); // 异步复制点到设备
    
        if (are_points_montgomery_form) { // 如果点是蒙哥马利形式
          // 将点从蒙哥马利形式转换
          CHK_IF_RETURN(mont::from_montgomery(d_allocated_points, nof_points, stream_points, d_allocated_points));
        }
        d_points = d_allocated_points; // 设置设备上点的指针
      } else { // 点已经在设备上
        if (are_points_montgomery_form) { // 如果点是蒙哥马利形式
          // 在设备上分配内存并转换点
          CHK_IF_RETURN(cudaMallocAsync(&d_allocated_points, sizeof(A) * nof_points, stream_points));
          CHK_IF_RETURN(mont::from_montgomery(points, nof_points, stream_points, d_allocated_points));
          d_points = d_allocated_points; // 设置设备上点的指针
        } else {
          d_points = points; // 直接使用已经在设备上的点
        }
      }
    
      cudaEvent_t event_points_uploaded; // 定义CUDA事件用于标记点上传完成
      if (stream_points) { // 如果创建了点上传流
        CHK_IF_RETURN(cudaEventCreateWithFlags(&event_points_uploaded, cudaEventDisableTiming)); // 创建CUDA事件，不启用计时
        CHK_IF_RETURN(cudaEventRecord(event_points_uploaded, stream_points)); // 记录点上传完成事件
      }
    
      P* buckets; // 指向桶的设备指针
      CHK_IF_RETURN(cudaMallocAsync(&buckets, sizeof(P) * (total_nof_buckets + nof_bms_in_batch), stream)); // 在设备上分配内存用于桶
    
      // 使用最大线程数启动桶初始化内核
      NUM_THREADS = 1 << 10; // 设置线程数为1024
      NUM_BLOCKS = (total_nof_buckets + nof_bms_in_batch + NUM_THREADS - 1) / NUM_THREADS; // 计算块数，确保所有桶都被处理
      initialize_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(buckets, total_nof_buckets + nof_bms_in_batch); // 启动桶初始化内核
    
      // 移除零桶（如果存在）
      unsigned smallest_bucket_index; // 存储最小的桶索引
      CHK_IF_RETURN(cudaMemcpyAsync(
        &smallest_bucket_index, single_bucket_indices, sizeof(unsigned), cudaMemcpyDeviceToHost, stream)); // 异步复制最小桶索引到主机
      // 可能零桶实际上是空的？这种情况下，zero_bucket_offset设置为0
      unsigned zero_bucket_offset = (smallest_bucket_index == 0) ? 1 : 0; // 根据最小桶索引设置零桶偏移量
    
      // 按桶大小排序
      unsigned h_nof_buckets_to_compute; // 主机上需要计算的桶数量
      CHK_IF_RETURN(cudaMemcpyAsync(
        &h_nof_buckets_to_compute, nof_buckets_to_compute, sizeof(unsigned), cudaMemcpyDeviceToHost, stream)); // 异步复制需要计算的桶数量到主机
      CHK_IF_RETURN(cudaFreeAsync(nof_buckets_to_compute, stream)); // 释放设备上的桶数量指针
      h_nof_buckets_to_compute -= zero_bucket_offset; // 根据零桶偏移量调整需要计算的桶数量
    
      unsigned* sorted_bucket_sizes; // 排序后的桶大小
      CHK_IF_RETURN(cudaMallocAsync(&sorted_bucket_sizes, sizeof(unsigned) * h_nof_buckets_to_compute, stream)); // 在设备上分配内存用于排序后的桶大小
      unsigned* sorted_bucket_offsets; // 排序后的桶偏移量
      CHK_IF_RETURN(cudaMallocAsync(&sorted_bucket_offsets, sizeof(unsigned) * h_nof_buckets_to_compute, stream)); // 在设备上分配内存用于排序后的桶偏移量
      unsigned* sort_offsets_temp_storage{}; // 临时存储空间用于排序桶偏移量
      size_t sort_offsets_temp_storage_bytes = 0; // 临时存储空间的字节大小
      // 使用CUB的基数排序按降序排序桶大小和桶偏移量
      CHK_IF_RETURN(cub::DeviceRadixSort::SortPairsDescending(
        sort_offsets_temp_storage, sort_offsets_temp_storage_bytes, bucket_sizes + zero_bucket_offset,
        sorted_bucket_sizes, bucket_offsets + zero_bucket_offset, sorted_bucket_offsets, h_nof_buckets_to_compute, 0,
        sizeof(unsigned) * 8, stream));
      CHK_IF_RETURN(cudaMallocAsync(&sort_offsets_temp_storage, sort_offsets_temp_storage_bytes, stream)); // 分配临时存储空间
      // 再次调用排序内核进行实际排序
      CHK_IF_RETURN(cub::DeviceRadixSort::SortPairsDescending(
        sort_offsets_temp_storage, sort_offsets_temp_storage_bytes, bucket_sizes + zero_bucket_offset,
        sorted_bucket_sizes, bucket_offsets + zero_bucket_offset, sorted_bucket_offsets, h_nof_buckets_to_compute, 0,
        sizeof(unsigned) * 8, stream));
      CHK_IF_RETURN(cudaFreeAsync(sort_offsets_temp_storage, stream)); // 释放临时存储空间
      CHK_IF_RETURN(cudaFreeAsync(bucket_offsets, stream)); // 释放原始桶偏移量
    
      unsigned* sorted_single_bucket_indices; // 排序后的单个桶索引
      CHK_IF_RETURN(
        cudaMallocAsync(&sorted_single_bucket_indices, sizeof(unsigned) * h_nof_buckets_to_compute, stream)); // 在设备上分配内存
      unsigned* sort_single_temp_storage{}; // 临时存储空间用于排序单个桶索引
      size_t sort_single_temp_storage_bytes = 0; // 临时存储空间的字节大小
      /**  源代码
       // TODO: 可能有更优化的排序方式
      // 使用CUB的基数排序按降序排序单个桶索引
      CHK_IF_RETURN(cub::DeviceRadixSort::SortPairsDescending(
        sort_single_temp_storage, sort_single_temp_storage_bytes, bucket_sizes + zero_bucket_offset,
        sorted_bucket_sizes, single_bucket_indices + zero_bucket_offset, sorted_single_bucket_indices,
        h_nof_buckets_to_compute, 0, sizeof(unsigned) * 8, stream));
      CHK_IF_RETURN(cudaMallocAsync(&sort_single_temp_storage, sort_single_temp_storage_bytes, stream)); // 分配临时存储空间
      // 再次调用排序内核进行实际排序
      CHK_IF_RETURN(cub::DeviceRadixSort::SortPairsDescending(
        sort_single_temp_storage, sort_single_temp_storage_bytes, bucket_sizes + zero_bucket_offset,
        sorted_bucket_sizes, single_bucket_indices + zero_bucket_offset, sorted_single_bucket_indices,
        h_nof_buckets_to_compute, 0, sizeof(unsigned) * 8, stream));
      CHK_IF_RETURN(cudaFreeAsync(sort_single_temp_storage, stream)); // 释放临时存储空间
      CHK_IF_RETURN(cudaFreeAsync(bucket_sizes, stream)); // 释放桶大小
      CHK_IF_RETURN(cudaFreeAsync(single_bucket_indices, stream)); // 释放单个桶索引
      ***/
      // 使用混合排序策略优化桶排序
      // ————————————————————————————更新后————————————————————————————
      // 1. 对于小规模数据(<=1024)使用并行的bitonic排序
      // 2. 对于中等规模数据(<=65536)使用thrust::sort
      // 3. 对于大规模数据使用CUB的基数排序
      if (h_nof_buckets_to_compute <= 65536) {
        // 使用thrust::sort,适合中等规模数据
        thrust::sort_by_key(
          thrust::cuda::par.on(stream),
          bucket_sizes + zero_bucket_offset,
          bucket_sizes + zero_bucket_offset + h_nof_buckets_to_compute,
          single_bucket_indices + zero_bucket_offset,
          thrust::greater<unsigned>());
        
        // 复制排序结果
        CHK_IF_RETURN(cudaMemcpyAsync(
          sorted_bucket_sizes,
          bucket_sizes + zero_bucket_offset,
          sizeof(unsigned) * h_nof_buckets_to_compute,
          cudaMemcpyDeviceToDevice,
          stream));
        CHK_IF_RETURN(cudaMemcpyAsync(
          sorted_single_bucket_indices,
          single_bucket_indices + zero_bucket_offset,
          sizeof(unsigned) * h_nof_buckets_to_compute,
          cudaMemcpyDeviceToDevice,
          stream));
      }
      else {
        // 大规模数据使用CUB的基数排序
        CHK_IF_RETURN(cub::DeviceRadixSort::SortPairsDescending(
          sort_single_temp_storage,
          sort_single_temp_storage_bytes,
          bucket_sizes + zero_bucket_offset,
          sorted_bucket_sizes,
          single_bucket_indices + zero_bucket_offset,
          sorted_single_bucket_indices,
          h_nof_buckets_to_compute,
          0,
          sizeof(unsigned) * 8,
          stream));
        CHK_IF_RETURN(cudaMallocAsync(&sort_single_temp_storage, sort_single_temp_storage_bytes, stream));
        CHK_IF_RETURN(cub::DeviceRadixSort::SortPairsDescending(
          sort_single_temp_storage,
          sort_single_temp_storage_bytes,
          bucket_sizes + zero_bucket_offset,
          sorted_bucket_sizes,
          single_bucket_indices + zero_bucket_offset,
          sorted_single_bucket_indices,
          h_nof_buckets_to_compute,
          0,
          sizeof(unsigned) * 8,
          stream));
      }

      // 清理内存
      if (sort_single_temp_storage) {
        CHK_IF_RETURN(cudaFreeAsync(sort_single_temp_storage, stream));
      }
      CHK_IF_RETURN(cudaFreeAsync(bucket_sizes, stream));
      CHK_IF_RETURN(cudaFreeAsync(single_bucket_indices, stream));
      /// ————————————————————————————更新后————————————————————————————
      // find large buckets
      // 计算平均桶大小
      unsigned average_bucket_size = (single_msm_size / (1 << c)) * precompute_factor;
      // 确定一个桶必须多大才能被认为是一个"大桶"
      unsigned bucket_th = large_bucket_factor * average_bucket_size;
      unsigned* nof_large_buckets;
      CHK_IF_RETURN(cudaMallocAsync(&nof_large_buckets, sizeof(unsigned), stream));
      CHK_IF_RETURN(cudaMemset(nof_large_buckets, 0, sizeof(unsigned)));

      // 设置线程数和块数以适应设备
      unsigned TOTAL_THREADS = 163840; // 针对V100优化: 80 SMs * 2048 threads per SM
      unsigned cutoff_run_length = max(2, h_nof_buckets_to_compute / TOTAL_THREADS);
      unsigned cutoff_nof_runs = (h_nof_buckets_to_compute + cutoff_run_length - 1) / cutoff_run_length;
      NUM_THREADS = 1 << 5;
      NUM_BLOCKS = (cutoff_nof_runs + NUM_THREADS - 1) / NUM_THREADS;
      // 如果有足够的桶且阈值大于0，则启动内核来查找大桶
      if (h_nof_buckets_to_compute > 0 && bucket_th > 0)
        find_cutoff_kernel<S><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
          sorted_bucket_sizes, h_nof_buckets_to_compute, bucket_th, cutoff_run_length, nof_large_buckets);
      unsigned h_nof_large_buckets;
      CHK_IF_RETURN(
        cudaMemcpyAsync(&h_nof_large_buckets, nof_large_buckets, sizeof(unsigned), cudaMemcpyDeviceToHost, stream));
      CHK_IF_RETURN(cudaFreeAsync(nof_large_buckets, stream));

      if (stream_points) {
        // 到这里，点需要已经上传并去蒙哥马利化
        CHK_IF_RETURN(cudaStreamWaitEvent(stream, event_points_uploaded));
        CHK_IF_RETURN(cudaEventDestroy(event_points_uploaded));
        CHK_IF_RETURN(cudaStreamDestroy(stream_points));
      }

      cudaStream_t stream_large_buckets;
      cudaEvent_t event_large_buckets_accumulated;
      // ---------------- 这是处理大桶的开始（如果有大桶） -------------
      if (h_nof_large_buckets > 0 && bucket_th > 0) {
        CHK_IF_RETURN(cudaStreamCreate(&stream_large_buckets));
        CHK_IF_RETURN(cudaEventCreateWithFlags(&event_large_buckets_accumulated, cudaEventDisableTiming));

        unsigned* sorted_bucket_sizes_sum;
        CHK_IF_RETURN(cudaMallocAsync(
          &sorted_bucket_sizes_sum, sizeof(unsigned) * (h_nof_large_buckets + 1), stream_large_buckets));
        CHK_IF_RETURN(cudaMemsetAsync(sorted_bucket_sizes_sum, 0, sizeof(unsigned), stream_large_buckets));
        unsigned* large_bucket_temp_storage{};
        size_t large_bucket_temp_storage_bytes = 0;
        CHK_IF_RETURN(cub::DeviceScan::InclusiveSum(
          large_bucket_temp_storage, large_bucket_temp_storage_bytes, sorted_bucket_sizes, sorted_bucket_sizes_sum + 1,
          h_nof_large_buckets, stream_large_buckets));
        CHK_IF_RETURN(
          cudaMallocAsync(&large_bucket_temp_storage, large_bucket_temp_storage_bytes, stream_large_buckets));
        CHK_IF_RETURN(cub::DeviceScan::InclusiveSum(
          large_bucket_temp_storage, large_bucket_temp_storage_bytes, sorted_bucket_sizes, sorted_bucket_sizes_sum + 1,
          h_nof_large_buckets, stream_large_buckets));
        CHK_IF_RETURN(cudaFreeAsync(large_bucket_temp_storage, stream_large_buckets));
        unsigned h_nof_pts_in_large_buckets;
        CHK_IF_RETURN(cudaMemcpyAsync(
          &h_nof_pts_in_large_buckets, sorted_bucket_sizes_sum + h_nof_large_buckets, sizeof(unsigned),
          cudaMemcpyDeviceToHost, stream_large_buckets));
        unsigned h_largest_bucket;
        CHK_IF_RETURN(cudaMemcpyAsync(
          &h_largest_bucket, sorted_bucket_sizes, sizeof(unsigned), cudaMemcpyDeviceToHost, stream_large_buckets));

        // 计算大桶所需的线程数
        // 公式解释:
        // 1. h_nof_pts_in_large_buckets/average_bucket_size 计算基本需要的线程数
        // 2. +h_nof_large_buckets 添加额外线程以处理不能被平均大小整除的桶
        unsigned large_buckets_nof_threads =
          (h_nof_pts_in_large_buckets + average_bucket_size - 1) / average_bucket_size + h_nof_large_buckets;
        // 计算大桶数量的对数值，用于位操作
        unsigned log_nof_large_buckets = (unsigned)ceil(std::log2(h_nof_large_buckets));

        // 分配大桶索引数组内存
        unsigned* large_bucket_indices;
        CHK_IF_RETURN(cudaMallocAsync(&large_bucket_indices, sizeof(unsigned) * large_buckets_nof_threads, stream));

        // 配置并启动大桶索引初始化内核
        NUM_THREADS = max(1, min(1 << 8, h_nof_large_buckets));
        NUM_BLOCKS = (h_nof_large_buckets + NUM_THREADS - 1) / NUM_THREADS;
        initialize_large_bucket_indices<P><<<NUM_BLOCKS, NUM_THREADS, 0, stream_large_buckets>>>(
          sorted_bucket_sizes_sum, average_bucket_size, h_nof_large_buckets, log_nof_large_buckets,
          large_bucket_indices);

        // 分配大桶数组内存
        P* large_buckets;
        CHK_IF_RETURN(cudaMallocAsync(&large_buckets, sizeof(P) * large_buckets_nof_threads, stream_large_buckets));

        // 配置并启动大桶累加内核
        // 这个内核将点累加到相应的大桶中
        NUM_THREADS = max(1, min(1 << 8, large_buckets_nof_threads));
        NUM_BLOCKS = (large_buckets_nof_threads + NUM_THREADS - 1) / NUM_THREADS;
        accumulate_large_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream_large_buckets>>>(
          large_buckets, sorted_bucket_offsets, sorted_bucket_sizes, large_bucket_indices, sorted_point_indices,
          d_points, h_nof_large_buckets, c, average_bucket_size, log_nof_large_buckets, large_buckets_nof_threads);

        // 配置并启动桶大小归一化内核
        // 这步是必要的，因为前面的归约操作改变了桶的大小和偏移
        NUM_THREADS = max(1, min(MAX_TH, h_nof_large_buckets));
        NUM_BLOCKS = (h_nof_large_buckets + NUM_THREADS - 1) / NUM_THREADS;
        // normalization is needed to update buckets sizes and offsets due to reduction that already took place
        normalize_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream_large_buckets>>>(
          sorted_bucket_sizes_sum, average_bucket_size, h_nof_large_buckets);
        // reduce
        for (int s = h_largest_bucket; s > 1; s = ((s + 1) >> 1)) {
          // 首先归一化桶大小
          NUM_THREADS = max(1, min(MAX_TH, h_nof_large_buckets));
          NUM_BLOCKS = (h_nof_large_buckets + NUM_THREADS - 1) / NUM_THREADS;
          normalize_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream_large_buckets>>>(
            sorted_bucket_sizes, s == h_largest_bucket ? average_bucket_size : 2, h_nof_large_buckets);
          
          // 然后执行可变大小的和归约
          NUM_THREADS = max(1, min(MAX_TH, large_buckets_nof_threads));
          NUM_BLOCKS = (large_buckets_nof_threads + NUM_THREADS - 1) / NUM_THREADS;
          sum_reduction_variable_size_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream_large_buckets>>>(
            large_buckets, sorted_bucket_sizes_sum, sorted_bucket_sizes, large_bucket_indices,
            large_buckets_nof_threads);
        }

        // 释放大桶索引内存
        CHK_IF_RETURN(cudaFreeAsync(large_bucket_indices, stream_large_buckets));

        // 配置并启动分发内核，将归约后的大桶结果分发到最终的桶数组中
        NUM_THREADS = max(1, min(MAX_TH, h_nof_large_buckets));
        NUM_BLOCKS = (h_nof_large_buckets + NUM_THREADS - 1) / NUM_THREADS;
        distribute_large_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream_large_buckets>>>(
          large_buckets, buckets, sorted_bucket_sizes_sum, sorted_single_bucket_indices, h_nof_large_buckets,
          nof_buckets + nof_bms_per_msm, c + bm_bitsize);

        // 清理大桶相关内存
        CHK_IF_RETURN(cudaFreeAsync(large_buckets, stream_large_buckets));
        CHK_IF_RETURN(cudaFreeAsync(sorted_bucket_sizes_sum, stream_large_buckets));

        // 记录大桶处理完成事件
        CHK_IF_RETURN(cudaEventRecord(event_large_buckets_accumulated, stream_large_buckets));
      }

      // ------------------------- Accumulation of (non-large) buckets ---------------------------------
      if (h_nof_buckets_to_compute > h_nof_large_buckets) {
        NUM_THREADS = 1 << 8; // 设置线程数为256
        NUM_BLOCKS = (h_nof_buckets_to_compute - h_nof_large_buckets + NUM_THREADS - 1) / NUM_THREADS; // 计算块数，确保所有桶都被处理
        // 启动累加非大型桶的内核，使用配置好的线程和块数
        accumulate_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
          buckets, 
          sorted_bucket_offsets + h_nof_large_buckets, // 非大型桶的偏移量
          sorted_bucket_sizes + h_nof_large_buckets,   // 非大型桶的大小
          sorted_single_bucket_indices + h_nof_large_buckets, // 非大型桶的索引
          sorted_point_indices, // 点的索引
          d_points, // EC 点
          nof_buckets + nof_bms_per_msm, // 总桶数加上每个 MSM 的桶模块数
          h_nof_buckets_to_compute - h_nof_large_buckets, // 需要计算的非大型桶数量
          c + bm_bitsize, // 位移量
          c // 原始位数
        );
      }

      // 释放排序后的点索引、桶大小、桶偏移和单个桶索引的设备内存
      CHK_IF_RETURN(cudaFreeAsync(sorted_point_indices, stream));
      CHK_IF_RETURN(cudaFreeAsync(sorted_bucket_sizes, stream));
      CHK_IF_RETURN(cudaFreeAsync(sorted_bucket_offsets, stream));
      CHK_IF_RETURN(cudaFreeAsync(sorted_single_bucket_indices, stream));

      // 如果存在大型桶且桶阈值大于0，则等待大型桶处理完成
      if (h_nof_large_buckets > 0 && bucket_th > 0) {
        // 等待大型桶处理完成的事件
        CHK_IF_RETURN(cudaStreamWaitEvent(stream, event_large_buckets_accumulated));
        // 销毁用于大型桶处理的CUDA流
        CHK_IF_RETURN(cudaStreamDestroy(stream_large_buckets));
      }

      P* d_allocated_final_result = nullptr; // 指向最终结果的设备指针
      // 如果结果不在设备上，分配用于存储最终结果的设备内存
      if (!are_results_on_device)
        CHK_IF_RETURN(cudaMallocAsync(&d_allocated_final_result, sizeof(P) * batch_size, stream));

      // --- 桶的归约操作在这里进行，之后每个桶模块/窗口将得到一个单一的和 ---
      unsigned nof_final_results_per_msm = nof_bms_per_msm; // 对于大三角累加，这是每个 MSM 的桶模块数量
      P* final_results;
      if (is_big_triangle || c == 1) {
        // 如果使用大三角累加或位数为1，分配最终结果的内存
        CHK_IF_RETURN(cudaMallocAsync(&final_results, sizeof(P) * nof_bms_in_batch, stream));
        // 启动桶模块求和内核，每个桶模块由一个线程处理
        NUM_THREADS = 128; // 设置线程数为32
        NUM_BLOCKS = (nof_bms_in_batch + NUM_THREADS - 1) / NUM_THREADS; // 计算块数
        optimized_big_triangle_sum_kernel<<<NUM_BLOCKS, NUM_THREADS, sizeof(P) * NUM_THREADS, stream>>>(
          buckets, // 输入的桶数组
          final_results, // 输出的最终结果数组
          nof_bms_in_batch, // 桶模块的数量
          c // 位数
        );
      } else {
        // 如果不使用大三角累加，采用迭代归约算法，该算法可以在并行流上运行两种类型的归约
        cudaStream_t stream_reduction;
        cudaEvent_t event_finished_reduction;
        // 创建用于归约的CUDA流
        CHK_IF_RETURN(cudaStreamCreate(&stream_reduction));
        // 创建用于标记归约完成的事件
        CHK_IF_RETURN(cudaEventCreateWithFlags(&event_finished_reduction, cudaEventDisableTiming));

        unsigned source_bits_count = c; // 源位数
        unsigned source_windows_count = nof_bms_per_msm; // 源窗口数量
        unsigned source_buckets_count = nof_buckets + nof_bms_per_msm; // 每个 MSM 包含的桶数，包括零桶
        unsigned target_windows_count;
        P* source_buckets = buckets; // 源桶数组
        buckets = nullptr; // 清空源桶指针
        P* target_buckets; // 目标桶数组
        P* temp_buckets1; // 临时桶数组1，用于类型1归约（交错，底层窗口 - 偶数）
        P* temp_buckets2; // 临时桶数组2，用于类型2归约（串行，上层窗口 - 奇数）
        for (unsigned i = 0;; i++) {
          // 计算目标位数为源位数的一半，向上取整
          const unsigned target_bits_count = (source_bits_count + 1) >> 1; 
          // 目标窗口数量为源窗口数量的两倍
          target_windows_count = source_windows_count << 1; 
          // 计算目标桶的总数
          const unsigned target_buckets_count = target_windows_count << target_bits_count; 
          // 为目标桶分配内存
          CHK_IF_RETURN(cudaMallocAsync(&target_buckets, sizeof(P) * target_buckets_count * batch_size, stream));
          // 为类型1归约（交错，底层窗口 - 偶数）分配临时桶数组1的内存
          CHK_IF_RETURN(cudaMallocAsync(
            &temp_buckets1, sizeof(P) * source_buckets_count * batch_size,
            stream));
          // 为类型2归约（串行，上层窗口 - 奇数）分配临时桶数组2的内存
          CHK_IF_RETURN(cudaMallocAsync(
            &temp_buckets2, sizeof(P) * source_buckets_count * batch_size,
            stream));
          // 初始化目标桶数组，确保在奇数c的情况下需要初始化
          initialize_buckets_kernel<<<(target_buckets_count * batch_size + 255) / 256, 256>>>(
            target_buckets, target_buckets_count * batch_size); // initialization is needed for the odd c case

          for (unsigned j = 0; j < target_bits_count; j++) {
            const bool is_first_iter = (j == 0); // 判断是否为第一次迭代
            const bool is_second_iter = (j == 1); // 判断是否为第二次迭代
            const bool is_last_iter = (j == target_bits_count - 1); // 判断是否为最后一次迭代
            const bool is_odd_c = source_bits_count & 1; // 判断源位数是否为奇数

            // 计算本次归约需要的线程数
            unsigned nof_threads =
              (((source_windows_count << target_bits_count) - source_windows_count) << (target_bits_count - 1 - j)) *
              batch_size; // 计算需要归约的部分数量（减去要排除的部分）并乘以每部分需要的线程数
            NUM_THREADS = max(1, min(MAX_TH, nof_threads)); // 限制线程数在1到MAX_TH之间
            NUM_BLOCKS = (nof_threads + NUM_THREADS - 1) / NUM_THREADS; // 计算块数
            if (!is_odd_c || !is_first_iter) { // 如果c不是奇数或不是第一次迭代，则执行以下归约操作
              single_stage_multi_reduction_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
                is_first_iter || (is_second_iter && is_odd_c) ? source_buckets : temp_buckets1, // 根据条件选择源桶
                is_last_iter ? target_buckets : temp_buckets1, // 如果是最后一次迭代，目标桶为target_buckets，否则为temp_buckets1
                1 << source_bits_count, // 源位数的2次幂
                1 << (source_bits_count - j + (is_odd_c ? 1 : 0)), // 根据当前迭代调整的位数
                is_last_iter ? 1 << target_bits_count : 0, // 如果是最后一次迭代，设置写入索引
                1 << target_bits_count, // 写入步幅
                0 /*=write_phase*/, // 写入阶段标志
                (1 << target_bits_count) - 1, // 写入掩码
                nof_threads // 线程数量
              );
            }

            // 重新计算线程数，用于第二个归约阶段
            nof_threads =
              (((source_windows_count << (source_bits_count - target_bits_count)) - source_windows_count)
               << (target_bits_count - 1 - j)) *
              batch_size; // 计算需要归约的部分数量并乘以每部分需要的线程数
            NUM_THREADS = max(1, min(MAX_TH, nof_threads)); // 限制线程数在1到MAX_TH之间
            NUM_BLOCKS = (nof_threads + NUM_THREADS - 1) / NUM_THREADS; // 计算块数
            // 启动第二种类型的归约内核，处理不同的归约逻辑
            single_stage_multi_reduction_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream_reduction>>>(
              is_first_iter ? source_buckets : temp_buckets2, // 如果是第一次迭代，使用source_buckets作为源，否则使用temp_buckets2
              is_last_iter ? target_buckets : temp_buckets2, // 如果是最后一次迭代，目标桶为target_buckets，否则为temp_buckets2
              1 << target_bits_count, // 目标位数的2次幂
              1 << (target_bits_count - j), // 根据当前迭代调整的位数
              is_last_iter ? 1 << target_bits_count : 0, // 如果是最后一次迭代，设置写入索引
              1 << (target_bits_count - (is_odd_c ? 1 : 0)), // 写入步幅，根据c的奇偶性调整
              1 /*=write_phase*/, // 写入阶段标志
              (1 << (target_bits_count - (is_odd_c ? 1 : 0))) - 1, // 写入掩码
              nof_threads // 线程数量
            );
          }
          CHK_IF_RETURN(cudaEventRecord(event_finished_reduction, stream_reduction));
          CHK_IF_RETURN(
            cudaStreamWaitEvent(stream, event_finished_reduction)); // sync streams after every write to target_buckets
          if (target_bits_count == 1) {
            // 注释：
                // 归约过程最终会为每个批处理元素生成 'target_windows_count' 个窗口。
                // 当 target_windows_count > bitsize 时，有些窗口会被保证为空。
                // 例如，考虑 bitsize=253 和 c=2。归约过程会生成 254 个桶模块（bms），
                // 但最显著的一个保证为零，因为标量的位数为 253。
                // 预计算和奇数 c 可能导致额外的空窗口。

            nof_final_results_per_msm = min(c * nof_bms_per_msm, bitsize);
            nof_bms_per_msm = target_windows_count;
            unsigned total_nof_final_results = nof_final_results_per_msm * batch_size;

            CHK_IF_RETURN(cudaMallocAsync(&final_results, sizeof(P) * total_nof_final_results, stream));

            // 对于V100 GPU，每个SM有64个CUDA核心，最佳线程数通常是32或128的倍数
            // 使用128线程可以更好地隐藏延迟并提高SM占用率
            NUM_THREADS = 128;
            NUM_BLOCKS = (total_nof_final_results + NUM_THREADS - 1) / NUM_THREADS;
            // 为最终结果分配设备内存
            // 启动最后一次归约内核，将目标桶中的值汇总到 final_results 中

            last_pass_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
              target_buckets, final_results, nof_final_results_per_msm, batch_size, nof_bms_per_msm, c);
            c = 1;
            CHK_IF_RETURN(cudaFreeAsync(source_buckets, stream));
            CHK_IF_RETURN(cudaFreeAsync(target_buckets, stream));
            CHK_IF_RETURN(cudaFreeAsync(temp_buckets1, stream));
            CHK_IF_RETURN(cudaFreeAsync(temp_buckets2, stream));
            CHK_IF_RETURN(cudaStreamDestroy(stream_reduction));
            break;
          }
          CHK_IF_RETURN(cudaFreeAsync(source_buckets, stream));
          CHK_IF_RETURN(cudaFreeAsync(temp_buckets1, stream));
          CHK_IF_RETURN(cudaFreeAsync(temp_buckets2, stream));
          // 将目标桶设置为新的源桶，为下一次归约迭代做准备
          source_buckets = target_buckets;
          target_buckets = nullptr;
          temp_buckets1 = nullptr;
          temp_buckets2 = nullptr;
          source_bits_count = target_bits_count;
          source_windows_count = target_windows_count;
          source_buckets_count = target_buckets_count;
        }
      }

      // ------- 这是最终阶段，桶模块/窗口的和将根据适当的权重进行加总 -------
      NUM_THREADS = 128; // 设置线程数为32
      NUM_BLOCKS = (batch_size + NUM_THREADS - 1) / NUM_THREADS; // 计算块数
      // 启动双倍加法内核，每个批次元素由一个线程处理
      final_accumulation_kernel<P, S><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
        final_results, 
        are_results_on_device ? final_result : d_allocated_final_result, // 如果结果在设备上，则直接使用，否则使用分配的设备内存
        batch_size, // 批次大小
        nof_final_results_per_msm, // 每个 MSM 的最终结果数量
        c // 位数
      );
      // 释放最终结果的设备内存
      CHK_IF_RETURN(cudaFreeAsync(final_results, stream));

      // 如果结果不在设备上，将结果从设备复制回主机
      if (!are_results_on_device)
        CHK_IF_RETURN(cudaMemcpyAsync(
          final_result, d_allocated_final_result, sizeof(P) * batch_size, cudaMemcpyDeviceToHost, stream));

      // 释放所有分配的设备内存
      if (d_allocated_scalars) CHK_IF_RETURN(cudaFreeAsync(d_allocated_scalars, stream));
      if (d_allocated_points) CHK_IF_RETURN(cudaFreeAsync(d_allocated_points, stream));
      if (d_allocated_final_result) CHK_IF_RETURN(cudaFreeAsync(d_allocated_final_result, stream));
      CHK_IF_RETURN(cudaFreeAsync(buckets, stream));

      // 如果不异步，等待所有CUDA操作完成
      if (!is_async) CHK_IF_RETURN(cudaStreamSynchronize(stream));

      return CHK_LAST(); // 返回最后的CUDA错误码
    }
  } // namespace

  template <typename S, typename A, typename P>
  cudaError_t msm(const S* scalars, const A* points, int msm_size, MSMConfig& config, P* results)
  {
    // 确定位数，如果配置中的 bitsize 为 0，则使用 S 的位数
    const int bitsize = (config.bitsize == 0) ? S::NBITS : config.bitsize;
    cudaStream_t& stream = config.ctx.stream; // 获取 CUDA 流

    // 获取最优的 c 值，如果配置中的 c 为 0，则调用 get_optimal_c 函数
    unsigned c = (config.c == 0) ? get_optimal_c(msm_size) : config.c;

    // 调用 bucket_method_msm 函数执行多标量乘法，并返回 CUDA 错误代码
    return CHK_STICKY(bucket_method_msm(
      bitsize, c, scalars, points, config.batch_size, msm_size,
      (config.points_size == 0) ? msm_size : config.points_size, results, config.are_scalars_on_device,
      config.are_scalars_montgomery_form, config.are_points_on_device, config.are_points_montgomery_form,
      config.are_results_on_device, config.is_big_triangle, config.large_bucket_factor, config.precompute_factor,
      config.is_async, stream));
  }

  template <typename A, typename P>
  cudaError_t precompute_msm_points(A* points, int msm_size, MSMConfig& config, A* output_points)
  {
    CHK_INIT_IF_RETURN(); // 初始化检查

    cudaStream_t& stream = config.ctx.stream; // 获取 CUDA 流
    unsigned c = (config.c == 0) ? get_optimal_c(msm_size) : config.c; // 获取 c 值

    // 异步复制输入点数组到输出数组
    CHK_IF_RETURN(cudaMemcpyAsync(
      output_points, points, sizeof(A) * config.points_size,
      config.are_points_on_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice, stream));

    // 计算每个桶的总数和位移
    unsigned total_nof_bms = (P::SCALAR_FF_NBITS - 1) / c + 1;
    unsigned shift = c * ((total_nof_bms - 1) / config.precompute_factor + 1);

    unsigned NUM_THREADS = 1 << 8; // 设置线程数
    unsigned NUM_BLOCKS = (config.points_size + NUM_THREADS - 1) / NUM_THREADS; // 计算块数
    // 对每个预计算因子进行左移操作
    for (int i = 1; i < config.precompute_factor; i++) {
      left_shift_kernel<A, P><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
        &output_points[(i - 1) * config.points_size], shift, config.points_size,
        &output_points[i * config.points_size]);
      }

    return CHK_LAST(); // 返回最后的 CUDA 错误代码
  }

  template <typename A, typename P>
  [[deprecated("Use precompute_msm_points instead.")]] cudaError_t precompute_msm_bases(
    A* bases,
    int bases_size,
    int precompute_factor,
    int _c,
    bool are_bases_on_device,
    device_context::DeviceContext& ctx,
    A* output_bases)
  {
    CHK_INIT_IF_RETURN(); // 初始化检查

    cudaStream_t& stream = ctx.stream; // 获取 CUDA 流

    // 异步复制输入基点数组到输出数组
    CHK_IF_RETURN(cudaMemcpyAsync(
      output_bases, bases, sizeof(A) * bases_size,
      are_bases_on_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice, stream));

    unsigned c = 16; // 设置 c 值为 16
    unsigned total_nof_bms = (P::SCALAR_FF_NBITS - 1) / c + 1; // 计算每个桶的总数
    unsigned shift = c * ((total_nof_bms - 1) / precompute_factor + 1); // 计算位移

    unsigned NUM_THREADS = 1 << 8; // 设置线程数
    unsigned NUM_BLOCKS = (bases_size + NUM_THREADS - 1) / NUM_THREADS; // 计算块数
    // 对每个预计算因子进行左移操作
    for (int i = 1; i < precompute_factor; i++) {
      left_shift_kernel<A, P><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
        &output_bases[(i - 1) * bases_size], shift, bases_size, &output_bases[i * bases_size]);
    }

    return CHK_LAST(); // 返回最后的 CUDA 错误代码
  }
} // namespace msm