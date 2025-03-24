build_msm:
	mkdir -p work
	nvcc -o work/test_msm -std=c++17 -arch=sm_80 -I. -I../../include tests/msm_test.cu

test_msm:
	mkdir -p work
	nvcc -o work2/test_msm -std=c++17 -arch=sm_70 -I.  -I../../include tests/msm_test.cu
	work/test_msm


# ./work/test_msm 10 1 1 1

# nvcc -o work_sort/test_msm -std=c++17 -arch=sm_70 -I.  -I../../include tests/msm_test.cu
mkdir -p work_triangle
nvcc -o work_triangle/test_msm -std=c++17 -arch=sm_70 -I.  -I../../include tests/msm_test.cu
nsys profile ./work_triangle/test_msm 17 1 1 15

mkdir -p work_threads
nvcc -o work_threads/test_msm -std=c++17 -arch=sm_70 -I.  -I../../include tests/msm_test.cu
nsys profile ./work_threads/test_msm 12 1 1 1



nvcc -o work_sort/test_msm -std=c++17 -arch=sm_70 \
     -I../../include \
     -L/usr/local/cuda-11.8/targets/x86_64-linux/lib \
     tests/msm_test.cu -lcuda -lcupti

ncu --metrics \
lts__t_sectors_op_read_hit.sum, \    # L2 读取命中次数
lts__t_sectors_op_read_miss.sum, \   # L2 读取未命中次数
lts__t_sectors_op_write_hit.sum \    # L2 写入命中次数
./work2/test_msm

ncu --csv --metrics sm__warps_active.avg.pct_of_peak_sustained_active ./work2/test_msm > report.csv

ldd ./work_sort/test_msm | grep cupti
cat /proc/driver/nvidia/version