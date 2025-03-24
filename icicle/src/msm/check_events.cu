#include <cupti.h>
#include <cuda_runtime.h>
#include <stdio.h>

// 定义要监控的缓存事件（Volta 架构专用）
#define L1_EVENT "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"   // L1 全局加载操作
#define L2_EVENT "lts__t_sectors_op_read_hit.sum"                   // L2 读取命中
// 注：Volta 架构中 L3 缓存由 GPU 显存实现，一般通过显存带宽工具监控

// 初始化 CUPTI 事件组
void setupCacheMonitoring(CUpti_EventGroup* group, const char* eventName) {
    CUpti_EventID eventId;
    CUcontext context;
    cuCtxGetCurrent(&context);  // 确保 CUDA 上下文存在
    
    // 获取事件 ID
    CUptiResult res = cuptiEventGetIdFromName(context, eventName, &eventId);
    if (res != CUPTI_SUCCESS) {
        printf("无法获取事件 %s\n", eventName);
        exit(1);
    }
    
    // 创建事件组并添加事件
    cuptiEventGroupCreate(context, group, 0);
    cuptiEventGroupAddEvent(*group, eventId);
    cuptiEventGroupEnable(*group);
}

__global__ void targetFunction(float* data) {
    // 目标函数：模拟缓存敏感操作
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx] = data[idx] * 2.0f;
}

int main() {
    float *d_data;
    cudaMalloc(&d_data, 1024 * sizeof(float));  // 分配显存
    
    // 初始化 CUPTI
    cuptiInit();
    
    // 监控 L1 缓存
    CUpti_EventGroup l1Group;
    setupCacheMonitoring(&l1Group, L1_EVENT);
    
    // 监控 L2 缓存
    CUpti_EventGroup l2Group;
    setupCacheMonitoring(&l2Group, L2_EVENT);
    
    // 启动内核并记录事件
    targetFunction<<<1, 1024>>>(d_data);
    cudaDeviceSynchronize();
    
    // 读取事件计数器
    uint64_t l1Count, l2Count;
    cuptiEventGroupReadEvent(l1Group, 0, &l1Count);
    cuptiEventGroupReadEvent(l2Group, 0, &l2Count);
    
    printf("L1 全局加载操作次数: %lu\n", l1Count);
    printf("L2 读取命中次数: %lu\n", l2Count);
    
    // 清理资源
    cuptiEventGroupDestroy(l1Group);
    cuptiEventGroupDestroy(l2Group);
    cudaFree(d_data);
    return 0;
}