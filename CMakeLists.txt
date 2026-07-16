# ====================================================================
# [AOT-WARMUP INTEGRATED HARDWARE RUNTIME SYSTEM - ULTRA-PRODUCTION V5.0]
# @file: CMakeLists.txt
# [PART 1/2]: Global Infrastructure Layout & Pybind11 Detection Lock
# ====================================================================
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(pim_hbm_bypass LANGUAGES CXX CUDA)

# [1] C++ 표준 규격 및 호스트 컴파일러 엄격한 정합성 설정
# [1] C++ Standard Specification & Strict Host Compiler Consistency Configuration
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# [2] 하드웨어 가속 컴파일러(NVCC) 전용 아키텍처 및 고속 연산 최적화 플래그 강제 고정
# [2] Enforce Architecture & High-Speed Mathematical Optimization Flags Dedicated to Hardware Accelerator Compiler (NVCC)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # 기본 사양으로 Ampere(sm_80) 및 Hopper(sm_90) 아키텍처 바이너리를 2중 빌드하도록 강제합니다.
    # Enforce dual-architecture binary compilation for Ampere (sm_80) and Hopper (sm_90) as the baseline specification.
    set(CMAKE_CUDA_ARCHITECTURES "80;90")
endif()

# [🛠️ V5.0 빌드 문자열 규격 교정]: 세미콜론 분기 꼬임 방지를 위해 단일 융합 플래그 문자열로 강제 할당
# [🛠️ V5.0 Build String Specification Correction]: Enforce allocation as a single unified flag string to prevent semicolon bifurcation wrapping anomalies.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -Xcompiler -fPIC")

# [3] 호스트 환경 내 pybind11 인프라 자동 감지 및 패키지 락
# [3] Automated Detection & Package Lock of pybind11 Infrastructure within the Host Environment
execute_process(
    COMMAND python3 -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)


# ====================================================================
# [AOT-WARMUP INTEGRATED HARDWARE RUNTIME SYSTEM - ULTRA-PRODUCTION V5.0]
# @file: CMakeLists.txt
# [PART 2/2]: Pybind11 Module Targeting & Final Silicon Linker Closure
# ====================================================================

if(NOT pybind11_DIR)
    message(FATAL_ERROR "[BUILD_FAILURE] 시스템 환경 내 pybind11 패키지를 발견하지 못했습니다. 'pip install pybind11'을 가동하십시오.")
endif()

find_package(pybind11 REQUIRED)

# PIM-HBM 코어 실리콘 링킹 타겟 정의 및 공유 라이브러리(.so) 속성 지정
# 파이썬에서 'import pim_hbm_bridge_core'로 직통 유입되도록 라이브러리 이름을 바인딩합니다.
# Define PIM-HBM Core Silicon Linking Target & Specify Shared Library (.so) Attributes
# Bind the library name to enable direct ingestion via 'import pim_hbm_bridge_core' in Python.
pybind11_add_module(pim_hbm_bridge_core MODULE pim_hbm_core.cu)

# 컴파일러가 CUDA 런타임 내장 프리미티브를 손실 없이 읽도록 바인딩
# [🛠️ V5.0 링커 보장]: NVCC 기반 링킹이 강제 락킹되도록 LINKER_LANGUAGE CUDA 명세 주입
# Bind the Compiler to Read CUDA Runtime Built-in Primitives without Precision Loss
# [🛠️ V5.0 Linker Guarantee]: Inject LINKER_LANGUAGE CUDA Specification to Enforce NVCC-based Linking Lock
set_target_properties(pim_hbm_bridge_core PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    LINKER_LANGUAGE CUDA
    PREFIX ""
    OUTPUT_NAME "pim_hbm_bridge_core"
)


# 헤더 포함 경로 명시화 방화벽 가동
# Activate Header Include Directories Explicit Specification Firewall
target_include_directories(pim_hbm_bridge_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

# 최종 시스템 링킹 마감 리포트 출력
# Output Final System Linking Closure Report
message(STATUS "====================================================================")
message(STATUS "🎯 PIM-HBM HARDWARE CO-DESIGN ENGINE CMAKE BUILD CONFIGURATION V5.0")
message(STATUS " - Target CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS " - Pybind11 Directory Found:  ${pybind11_DIR}")
message(STATUS "====================================================================")

