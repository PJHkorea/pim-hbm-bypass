/**
 * @file pim_hbm_complete_system.cu
 
 * [통합 빌드 지침]
 * Compile: nvcc -O3 -shared -Xcompiler -fPIC -arch=sm_80 $(python3 -m pybind11 --includes) pim_hbm_complete_system.cu -o pim_hbm_bridge_core$(python3-config --extension-suffix)
 * Import in Python: import pim_hbm_bridge_core
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>

namespace py = pybind11;

// ====================================================================
// [PART 1/3] PIM-HBM Hardware Register & Bank Layout Specification
// ====================================================================
#define PIM_SATURATION_UPPER 999.0f

struct __align__(32) PimMemoryCell32 {
    float param_w;          // 가중치 물리 상태 (Weight)
    float momentum_m;       // 1차 적률 레지스터 (Momentum)
    float variance_v;       // 2차 적률 레지스터 (Variance)
    float lr_mask;          // 학습률 제어 레일 (Learning Rate)
    float wd_mask;          // 가중치 감쇠 제어 레일 (Weight Decay)
    uint32_t cell_status;   // 뱅크 무결성 검증 플래그
    uint32_t coordinate_id; // 하드웨어 핀 및 다이(Die) 좌표
    uint32_t reserved_pad;  // 32-Byte 물리 정렬용 잔여 패딩
};

static_assert(sizeof(PimMemoryCell32) == 32, "⚠ [CRITICAL] 32-byte alignment failed!");

struct PimValueSystemConfig {
    static constexpr bool enforce_rigid_honesty = true;
    static constexpr bool activate_zero_copy_bypass = true;
    static constexpr bool enable_thread_private_mapping = true; 
};

// ====================================================================
// [PART 2/3] Pure Branchless PIM Internal Mathematical Acceleration Kernel
// ====================================================================
__global__ void pim_pure_branchless_core_kernel(
    PimMemoryCell32* __restrict__ bank_array_ptr,
    size_t total_cells, float beta1, float beta2, float lr, float wd
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    
    size_t is_out_of_bound = (size_t)(idx >= total_cells);
    uintptr_t mask = -static_cast<intptr_t>(is_out_of_bound);
    size_t safe_idx = idx & (~mask);

    float w = bank_array_ptr[safe_idx].param_w;
    uint32_t w_bits = __float_as_int(w);
    
    // [리팩토링 완결] NaN 연산자 우선순위 교정
    bool is_nan = ((w_bits & 0x7F800000U) == 0x7F800000U) && ((w_bits & 0x007FFFFFU) != 0U);
    w_bits &= ~(-static_cast<int32_t>(is_nan));
    w = __int_as_float(w_bits);

    float m = __fmaf_rn(beta1, bank_array_ptr[safe_idx].momentum_m, (1.0f - beta1) * w);
    float v = __fmaf_rn(beta2, bank_array_ptr[safe_idx].variance_v, (1.0f - beta2) * w * w);
    float u = m * rsqrtf(v + 1e-9f);

    float update = u * lr + w * (wd * (1.0f - (float)is_out_of_bound));
    
    // [리팩토링 완결] 쓰기 단계 if 조건문 도살 완료 (해결책 1 스레드 분산)
    size_t private_dummy_idx = threadIdx.x % total_cells;
    size_t target_idx = (idx & (~mask)) | (private_dummy_idx & mask);

    float final_w = (w - update) * (1.0f - (float)is_out_of_bound) + w * (float)is_out_of_bound;
    float final_m = m * (1.0f - (float)is_out_of_bound) + bank_array_ptr[target_idx].momentum_m * (float)is_out_of_bound;
    float final_v = v * (1.0f - (float)is_out_of_bound) + bank_array_ptr[target_idx].variance_v * (float)is_out_of_bound;

    bank_array_ptr[target_idx].param_w   = final_w;
    bank_array_ptr[target_idx].momentum_m = final_m;
    bank_array_ptr[target_idx].variance_v = final_v;
}

// 외부 호스트 브리지용 C-래퍼 함수 구현
extern "C" void launch_pim_pure_branchless_core_kernel_host(
    void* bank_array_ptr, size_t total_cells, 
    float beta1, float beta2, float lr, float wd, cudaStream_t stream
) {
    int threads_per_block = 1024;
    size_t blocks_per_grid = (total_cells + threads_per_block - 1) / threads_per_block;
    
    pim_pure_branchless_core_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        reinterpret_cast<PimMemoryCell32*>(bank_array_ptr), 
        total_cells, beta1, beta2, lr, wd
    );
}

// ====================================================================
// [PART 3/3] 0ns Memory Copy Host Orchestrator & Python Ingestion Wrapper
// ====================================================================

py::array_t<float> ingest_pim_shared_memory_bypass(uintptr_t raw_device_ptr, size_t total_cells) {
    if (raw_device_ptr == 0) {
        throw std::invalid_argument("[PIM Bridge Failure] Null hardware pointer stream 유입.");
    }

    // PIM 반도체 뱅크에 상주하는 물리 가중치 레지스터(float)의 머리 주소를 직접 바인딩
    float* weight_head_ptr = reinterpret_cast<float*>(raw_device_ptr);

    // 파이썬 가비지 컬렉터(GC)의 메모리 해제 개입을 차단하는 0ns 보호 캡슐 구축
    py::capsule memory_fence(weight_head_ptr, [](void* p) {
        // [PASSIVE LOGGING]: 메모리 생명주기는 파이썬 소관이 아니므로 소멸자 실행 무력화
    });

    // 메모리 물리 복사를 완전히 박멸하고, 주소값 뷰만 JAX 텐서 공간으로 패스-스루 반환
    return py::array_t<float>(
        { total_cells },           // Shape: 1D 평탄화된 대규모 뱅크 셀 크기
        { sizeof(float) * 8 },     // Stride: PimMemoryCell32 구조체(32바이트) 간격으로 점프하며 weight만 스캔
        weight_head_ptr,
        memory_fence
    );
}

// 파이썬에서 import pim_hbm_bridge_core 로 불러올 모듈 정의
PYBIND11_MODULE(pim_hbm_bridge_core, m) {
    m.doc() = "5th-Gen Pure Algebraic PIM-HBM Hardware Interface Engine Core";
    m.def("ingest_pim_shared_memory_bypass", &ingest_pim_shared_memory_bypass, 
          "0ns 메모리 복사 오버헤드로 PIM 뱅크 주소선을 가로채는 바이패스 함수");
}
