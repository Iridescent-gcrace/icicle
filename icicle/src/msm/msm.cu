#include "msm/msm.cuh"

#include <cooperative_groups.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda.h>

#include <iostream>
#include <stdexcept>
#include <vector>

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
      if (tid >= nof_threads) return;

      // we need shifted tid because we don't want to be reducing into zero buckets, this allows to skip them.
      // for write_phase==1, the read pattern is different so we don't skip over anything.
      const int shifted_tid = write_phase ? tid : tid + (tid + step) / step;
      const int jump = block_size / 2;
      const int block_id = shifted_tid / jump;
      // here the reason for shifting is the same as for shifted_tid but we skip over entire blocks which happens
      // only for write_phase=1 because of its read pattern.
      const int shifted_block_id = write_phase ? block_id + (block_id + step) / step : block_id;
      const int block_tid = shifted_tid % jump;
      const unsigned read_ind = orig_block_size * shifted_block_id + block_tid;
      const unsigned write_ind = jump * shifted_block_id + block_tid;
      const unsigned v_r_key =
        write_stride ? ((write_ind / buckets_per_bm) * 2 + write_phase) * write_stride + write_ind % buckets_per_bm
                     : read_ind;
      v_r[v_r_key] = v[read_ind] + v[read_ind + jump];
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

    // 桶方法实现多标量乘法的核心函数
    template <typename S, typename P, typename A>
    cudaError_t bucket_method_msm(
      unsigned bitsize,        // 标量的位宽
      unsigned c,             // 每个窗口的位数
      const S* scalars,       // 输入标量数组
      const A* points,        // 输入点数组
      unsigned batch_size,    // 要计算的MSM批次数量
      unsigned single_msm_size, // 单个MSM的元素数量（即N）
      unsigned nof_points,    // 点数组中的总点数（共享或独立）
      P* final_result,         // 最终结果存储
      bool are_scalars_on_device,      // 标量是否已在设备
      bool are_scalars_montgomery_form, // 标量是否为蒙哥马利形式
      bool are_points_on_device,       // 点是否已在设备
      bool are_points_montgomery_form, // 点是否为蒙哥马利形式
      bool are_results_on_device,      // 结果是否存储在设备
      bool is_big_triangle,    // 是否使用大三角求和优化
      int large_bucket_factor, // 大桶判定因子（相对于平均桶大小）
      int precompute_factor,   // 预计算因子
      bool is_async,           // 是否异步执行
      cudaStream_t stream)     // CUDA流
    {
      CHK_INIT_IF_RETURN(); // 初始化错误检查

      // 计算总标量数并验证点数有效性
      const unsigned nof_scalars = batch_size * single_msm_size;
      const bool is_nof_points_valid = ((single_msm_size * batch_size) % nof_points == 0);
      if (!is_nof_points_valid) {
        THROW_ICICLE_ERR(IcicleError_t::InvalidArgument, 
          "bucket_method_msm: #points必须能被single_msm_size*batch_size整除");
      }

      // 处理标量数据（主机到设备拷贝及格式转换）
      const S* d_scalars;
      S* d_allocated_scalars = nullptr;
      if (!are_scalars_on_device) {
        // 分配设备内存并拷贝标量数据
        CHK_IF_RETURN(cudaMallocAsync(&d_allocated_scalars, sizeof(S) * nof_scalars, stream));
        CHK_IF_RETURN(cudaMemcpyAsync(d_allocated_scalars, scalars, sizeof(S) * nof_scalars, 
                                     cudaMemcpyHostToDevice, stream));

        // 如果需要，从蒙哥马利格式转换
        if (are_scalars_montgomery_form) {
          CHK_IF_RETURN(mont::from_montgomery(d_allocated_scalars, nof_scalars, stream, d_allocated_scalars));
        }
        d_scalars = d_allocated_scalars;
      } else { // 标量已在设备
        if (are_scalars_montgomery_form) { // 需要转换蒙哥马利格式
          CHK_IF_RETURN(cudaMallocAsync(&d_allocated_scalars, sizeof(S) * nof_scalars, stream));
          CHK_IF_RETURN(mont::from_montgomery(scalars, nof_scalars, stream, d_allocated_scalars));
          d_scalars = d_allocated_scalars;
        } else {
          d_scalars = scalars; // 直接使用设备指针
        }
      }

      // 计算桶模块相关参数
      unsigned total_bms_per_msm = (bitsize + c - 1) / c; // 每个MSM的总桶模块数
      unsigned nof_bms_per_msm = (total_bms_per_msm - 1) / precompute_factor + 1; // 考虑预计算后的桶模块数
      unsigned input_indexes_count = nof_scalars * total_bms_per_msm; // 总索引数
      unsigned bm_bitsize = (unsigned)ceil(std::log2(nof_bms_per_msm)); // 桶模块索引的位宽

      // 分配索引数组内存
      unsigned* bucket_indices;     // 桶索引数组
      unsigned* point_indices;      // 点索引数组
      unsigned* sorted_bucket_indices; // 排序后的桶索引
      unsigned* sorted_point_indices;  // 排序后的点索引
      CHK_IF_RETURN(cudaMallocAsync(&bucket_indices, sizeof(unsigned) * input_indexes_count, stream));
      CHK_IF_RETURN(cudaMallocAsync(&point_indices, sizeof(unsigned) * input_indexes_count, stream));
      CHK_IF_RETURN(cudaMallocAsync(&sorted_bucket_indices, sizeof(unsigned) * input_indexes_count, stream));
      CHK_IF_RETURN(cudaMallocAsync(&sorted_point_indices, sizeof(unsigned) * input_indexes_count, stream));

      // 将标量分割为数字窗口（启动内核）
      unsigned NUM_THREADS = 1 << 10; // 1024线程/块
      unsigned NUM_BLOCKS = (nof_scalars + NUM_THREADS - 1) / NUM_THREADS;
      split_scalars_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
        bucket_indices, point_indices, d_scalars, nof_scalars, nof_points, 
        single_msm_size, total_bms_per_msm, bm_bitsize, c, nof_bms_per_msm);
      nof_points *= precompute_factor; // 更新点数（考虑预计算）

      // ------------------------------ 标量排序阶段开始 ------------------------------
      // 使用CUB库对桶索引进行基数排序，将相同桶的标量分组
      unsigned* sort_indices_temp_storage{};
      size_t sort_indices_temp_storage_bytes;
      // 第一次调用获取临时存储大小
      CHK_IF_RETURN(cub::DeviceRadixSort::SortPairs(
        sort_indices_temp_storage, sort_indices_temp_storage_bytes, 
        bucket_indices, sorted_bucket_indices,
        point_indices, sorted_point_indices, 
        input_indexes_count, 0, sizeof(unsigned) * 8, stream));
      // 分配临时存储空间
      CHK_IF_RETURN(cudaMallocAsync(&sort_indices_temp_storage, sort_indices_temp_storage_bytes, stream));
      // 执行实际排序操作
      CHK_IF_RETURN(cub::DeviceRadixSort::SortPairs(
        sort_indices_temp_storage, sort_indices_temp_storage_bytes, 
        bucket_indices, sorted_bucket_indices,
        point_indices, sorted_point_indices, 
        input_indexes_count, 0, sizeof(unsigned) * 8, stream));
      // 释放临时存储和不再需要的数组
      CHK_IF_RETURN(cudaFreeAsync(sort_indices_temp_storage, stream));
      CHK_IF_RETURN(cudaFreeAsync(bucket_indices, stream));
      CHK_IF_RETURN(cudaFreeAsync(point_indices, stream));

      // 计算桶模块和桶数量
      unsigned nof_bms_in_batch = nof_bms_per_msm * batch_size; // 批次总桶模块数
      const unsigned nof_buckets = (nof_bms_per_msm << c) - nof_bms_per_msm; // 每个MSM的桶数（排除零桶）
      const unsigned total_nof_buckets = nof_buckets * batch_size; // 总桶数

      // 计算桶大小和分布
      unsigned* single_bucket_indices; // 唯一桶索引数组
      unsigned* bucket_sizes;          // 每个桶的大小
      unsigned* nof_buckets_to_compute; // 需要计算的桶数
      CHK_IF_RETURN(cudaMallocAsync(&single_bucket_indices, sizeof(unsigned) * (total_nof_buckets + 1), stream));
      CHK_IF_RETURN(cudaMallocAsync(&bucket_sizes, sizeof(unsigned) * (total_nof_buckets + 1), stream));
      CHK_IF_RETURN(cudaMallocAsync(&nof_buckets_to_compute, sizeof(unsigned), stream));
      
      // 使用CUB的运行长度编码统计桶信息
      unsigned* encode_temp_storage{};
      size_t encode_temp_storage_bytes = 0;
      CHK_IF_RETURN(cub::DeviceRunLengthEncode::Encode(
        encode_temp_storage, encode_temp_storage_bytes, 
        sorted_bucket_indices, single_bucket_indices, 
        bucket_sizes, nof_buckets_to_compute, 
        input_indexes_count, stream));
      CHK_IF_RETURN(cudaMallocAsync(&encode_temp_storage, encode_temp_storage_bytes, stream));
      CHK_IF_RETURN(cub::DeviceRunLengthEncode::Encode(
        encode_temp_storage, encode_temp_storage_bytes, 
        sorted_bucket_indices, single_bucket_indices, 
        bucket_sizes, nof_buckets_to_compute, 
        input_indexes_count, stream));
      CHK_IF_RETURN(cudaFreeAsync(encode_temp_storage, stream));
      CHK_IF_RETURN(cudaFreeAsync(sorted_bucket_indices, stream));

      // 计算桶的偏移量（每个桶在数组中的起始位置）
      unsigned* bucket_offsets; // 桶偏移数组
      CHK_IF_RETURN(cudaMallocAsync(&bucket_offsets, sizeof(unsigned) * (total_nof_buckets + 1), stream));
      unsigned* offsets_temp_storage{};
      size_t offsets_temp_storage_bytes = 0;
      // 使用CUB的独占扫描计算偏移
      CHK_IF_RETURN(cub::DeviceScan::ExclusiveSum(
        offsets_temp_storage, offsets_temp_storage_bytes, 
        bucket_sizes, bucket_offsets, 
        total_nof_buckets + 1, stream));
      CHK_IF_RETURN(cudaMallocAsync(&offsets_temp_storage, offsets_temp_storage_bytes, stream));
      CHK_IF_RETURN(cub::DeviceScan::ExclusiveSum(
        offsets_temp_storage, offsets_temp_storage_bytes, 
        bucket_sizes, bucket_offsets, 
        total_nof_buckets + 1, stream));
      CHK_IF_RETURN(cudaFreeAsync(offsets_temp_storage, stream));

      // ----------- 并行上传点数据（如果点在主机）------------
      const A* d_points; // 设备点指针
      A* d_allocated_points = nullptr; // 分配的设备内存
      cudaStream_t stream_points = nullptr; // 点处理专用流
      if (!are_points_on_device || are_points_montgomery_form) 
        CHK_IF_RETURN(cudaStreamCreate(&stream_points));
      
      if (!are_points_on_device) {
        // 主机到设备拷贝
        CHK_IF_RETURN(cudaMallocAsync(&d_allocated_points, sizeof(A) * nof_points, stream_points));
        CHK_IF_RETURN(cudaMemcpyAsync(d_allocated_points, points, sizeof(A) * nof_points, 
                                     cudaMemcpyHostToDevice, stream_points));

        // 蒙哥马利格式转换
        if (are_points_montgomery_form) {
          CHK_IF_RETURN(mont::from_montgomery(d_allocated_points, nof_points, stream_points, d_allocated_points));
        }
        d_points = d_allocated_points;
      } else { // 点已在设备
        if (are_points_montgomery_form) {
          CHK_IF_RETURN(cudaMallocAsync(&d_allocated_points, sizeof(A) * nof_points, stream_points));
          CHK_IF_RETURN(mont::from_montgomery(points, nof_points, stream_points, d_allocated_points));
          d_points = d_allocated_points;
        } else {
          d_points = points; // 直接使用设备指针
        }
      }

      // 创建事件用于同步点数据上传完成
      cudaEvent_t event_points_uploaded;
      if (stream_points) {
        CHK_IF_RETURN(cudaEventCreateWithFlags(&event_points_uploaded, cudaEventDisableTiming));
        CHK_IF_RETURN(cudaEventRecord(event_points_uploaded, stream_points));
      }

      // 初始化桶存储
      P* buckets; // 桶数据存储
      CHK_IF_RETURN(cudaMallocAsync(&buckets, sizeof(P) * (total_nof_buckets + nof_bms_in_batch), stream));
      
      // 启动桶初始化内核
      NUM_THREADS = 1 << 10; // 1024线程/块
      NUM_BLOCKS = (total_nof_buckets + nof_bms_in_batch + NUM_THREADS - 1) / NUM_THREADS;
      initialize_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
        buckets, total_nof_buckets + nof_bms_in_batch);

      // 处理零桶（可能为空的情况）
      unsigned smallest_bucket_index;
      CHK_IF_RETURN(cudaMemcpyAsync(&smallest_bucket_index, single_bucket_indices, 
                                   sizeof(unsigned), cudaMemcpyDeviceToHost, stream));
      unsigned zero_bucket_offset = (smallest_bucket_index == 0) ? 1 : 0; // 零桶偏移量

      // 按桶大小排序（降序）
      unsigned h_nof_buckets_to_compute; // 主机端需要计算的桶数
      CHK_IF_RETURN(cudaMemcpyAsync(&h_nof_buckets_to_compute, nof_buckets_to_compute, 
                                   sizeof(unsigned), cudaMemcpyDeviceToHost, stream));
      CHK_IF_RETURN(cudaFreeAsync(nof_buckets_to_compute, stream));
      h_nof_buckets_to_compute -= zero_bucket_offset; // 调整有效桶数

      // 分配排序后的桶大小和偏移数组
      unsigned* sorted_bucket_sizes;
      unsigned* sorted_bucket_offsets;
      CHK_IF_RETURN(cudaMallocAsync(&sorted_bucket_sizes, sizeof(unsigned) * h_nof_buckets_to_compute, stream));
      CHK_IF_RETURN(cudaMallocAsync(&sorted_bucket_offsets, sizeof(unsigned) * h_nof_buckets_to_compute, stream));
      
      // 使用CUB进行降序排序
      unsigned* sort_offsets_temp_storage{};
      size_t sort_offsets_temp_storage_bytes = 0;
      CHK_IF_RETURN(cub::DeviceRadixSort::SortPairsDescending(
        sort_offsets_temp_storage, sort_offsets_temp_storage_bytes, 
        bucket_sizes + zero_bucket_offset, sorted_bucket_sizes,
        bucket_offsets + zero_bucket_offset, sorted_bucket_offsets, 
        h_nof_buckets_to_compute, 0, sizeof(unsigned) * 8, stream));
      CHK_IF_RETURN(cudaMallocAsync(&sort_offsets_temp_storage, sort_offsets_temp_storage_bytes, stream));
      CHK_IF_RETURN(cub::DeviceRadixSort::SortPairsDescending(
        sort_offsets_temp_storage, sort_offsets_temp_storage_bytes, 
        bucket_sizes + zero_bucket_offset, sorted_bucket_sizes,
        bucket_offsets + zero_bucket_offset, sorted_bucket_offsets, 
        h_nof_buckets_to_compute, 0, sizeof(unsigned) * 8, stream));
      CHK_IF_RETURN(cudaFreeAsync(sort_offsets_temp_storage, stream));
      CHK_IF_RETURN(cudaFreeAsync(bucket_offsets, stream));

      // ----------------- 大桶检测和初始化 -----------------
      // 计算平均桶大小：总元素数除以桶数，再乘以预计算因子
      unsigned average_bucket_size = (single_msm_size / (1 << c)) * precompute_factor;
      // 设置大桶阈值：平均大小乘以大桶因子
      unsigned bucket_th = large_bucket_factor * average_bucket_size;
      // 分配设备内存存储大桶数量
      unsigned* nof_large_buckets;
      CHK_IF_RETURN(cudaMallocAsync(&nof_large_buckets, sizeof(unsigned), stream));
      CHK_IF_RETURN(cudaMemset(nof_large_buckets, 0, sizeof(unsigned)));

      // ----------------- 大桶检测内核配置 -----------------
      unsigned TOTAL_THREADS = 129000; // 设备相关的总线程数（TODO：需要根据设备动态调整）
      // 计算每个线程处理的桶数（至少2个）
      unsigned cutoff_run_length = max(2, h_nof_buckets_to_compute / TOTAL_THREADS);
      // 计算运行批次数（向上取整）
      unsigned cutoff_nof_runs = (h_nof_buckets_to_compute + cutoff_run_length - 1) / cutoff_run_length;
      NUM_THREADS = 1 << 5; // 每块32个线程
      NUM_BLOCKS = (cutoff_nof_runs + NUM_THREADS - 1) / NUM_THREADS;

      // 启动大桶检测内核
      if (h_nof_buckets_to_compute > 0 && bucket_th > 0)
        find_cutoff_kernel<S><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
          sorted_bucket_sizes, h_nof_buckets_to_compute, bucket_th, cutoff_run_length, nof_large_buckets);

      // 将大桶数量从设备复制到主机
      unsigned h_nof_large_buckets;
      CHK_IF_RETURN(cudaMemcpyAsync(&h_nof_large_buckets, nof_large_buckets, sizeof(unsigned), 
                    cudaMemcpyDeviceToHost, stream));
      CHK_IF_RETURN(cudaFreeAsync(nof_large_buckets, stream));

      // ----------------- 点数据同步处理 -----------------
      if (stream_points) {
        // 确保点数据已上传并完成蒙哥马利形式转换
        CHK_IF_RETURN(cudaStreamWaitEvent(stream, event_points_uploaded));
        CHK_IF_RETURN(cudaEventDestroy(event_points_uploaded));
        CHK_IF_RETURN(cudaStreamDestroy(stream_points));
      }

      // ----------------- 大桶处理流程 -----------------
      cudaStream_t stream_large_buckets;
      cudaEvent_t event_large_buckets_accumulated;
      if (h_nof_large_buckets > 0 && bucket_th > 0) {
        // 创建专用流和事件用于大桶处理
        CHK_IF_RETURN(cudaStreamCreate(&stream_large_buckets));
        CHK_IF_RETURN(cudaEventCreateWithFlags(&event_large_buckets_accumulated, cudaEventDisableTiming));

        // 计算大桶的累积和
        unsigned* sorted_bucket_sizes_sum;
        CHK_IF_RETURN(cudaMallocAsync(&sorted_bucket_sizes_sum, 
          sizeof(unsigned) * (h_nof_large_buckets + 1), stream_large_buckets));
        CHK_IF_RETURN(cudaMemsetAsync(sorted_bucket_sizes_sum, 0, sizeof(unsigned), stream_large_buckets));

        // 使用CUB库计算包含扫描（前缀和）
        unsigned* large_bucket_temp_storage{};
        size_t large_bucket_temp_storage_bytes = 0;
        // 第一次调用获取所需临时存储大小
        CHK_IF_RETURN(cub::DeviceScan::InclusiveSum(/*...*/));
        // 分配临时存储空间并执行实际的扫描操作
        CHK_IF_RETURN(cudaMallocAsync(&large_bucket_temp_storage, large_bucket_temp_storage_bytes, 
                      stream_large_buckets));
        CHK_IF_RETURN(cub::DeviceScan::InclusiveSum(/*...*/));

        // ----------------- 大桶归约和分发 -----------------
        // 计算大桶处理所需的线程数
        unsigned large_buckets_nof_threads =
          (h_nof_pts_in_large_buckets + average_bucket_size - 1) / average_bucket_size + h_nof_large_buckets;
        unsigned log_nof_large_buckets = (unsigned)ceil(std::log2(h_nof_large_buckets));
        
        // 初始化大桶索引
        unsigned* large_bucket_indices;
        CHK_IF_RETURN(cudaMallocAsync(&large_bucket_indices, sizeof(unsigned) * large_buckets_nof_threads, 
                      stream));

        // 多阶段归约循环
        for (int s = h_largest_bucket; s > 1; s = ((s + 1) >> 1)) {
          // 归约前标准化桶大小
          normalize_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream_large_buckets>>>(/*...*/);
          // 执行可变大小的和归约
          sum_reduction_variable_size_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream_large_buckets>>>(/*...*/);
        }

        // 分发大桶结果回主桶数组
        distribute_large_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream_large_buckets>>>(/*...*/);
      }

      // ----------------- 非大桶累加处理 -----------------
      if (h_nof_buckets_to_compute > h_nof_large_buckets) {
        // 处理剩余的常规桶
        accumulate_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(/*...*/);
      }

      // ----------------- 最终归约和结果计算 -----------------
      // 选择归约策略：大三角求和或迭代归约
      if (is_big_triangle || c == 1) {
        // 使用大三角求和方法
        big_triangle_sum_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(/*...*/);
      } else {
        // 使用迭代归约算法
        // 初始化源和目标缓冲区
        unsigned source_bits_count = c;
        unsigned source_windows_count = nof_bms_per_msm;
        
        // 迭代归约循环
        for (unsigned i = 0;; i++) {
          // 计算目标位数和窗口数
          const unsigned target_bits_count = (source_bits_count + 1) >> 1;
          target_windows_count = source_windows_count << 1;
          
          // 内部归约循环
          for (unsigned j = 0; j < target_bits_count; j++) {
            // 执行单阶段多重归约
            single_stage_multi_reduction_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(/*...*/);
          }
        }
      }

      // ----------------- 最终累加和清理 -----------------
      // 使用双倍加法算法计算最终结果
      final_accumulation_kernel<P, S><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(/*...*/);

      // 释放所有临时分配的内存
      if (d_allocated_scalars) CHK_IF_RETURN(cudaFreeAsync(d_allocated_scalars, stream));
      if (d_allocated_points) CHK_IF_RETURN(cudaFreeAsync(d_allocated_points, stream));
      if (d_allocated_final_result) CHK_IF_RETURN(cudaFreeAsync(d_allocated_final_result, stream));
      CHK_IF_RETURN(cudaFreeAsync(buckets, stream));

      // 如果不是异步模式，等待所有操作完成
      if (!is_async) CHK_IF_RETURN(cudaStreamSynchronize(stream));

      return CHK_LAST();
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