# ====================================================================
# [AOT-WARMUP INTEGRATED HARDWARE RUNTIME SYSTEM - PRODUCTION CMAKE]
# @file: CMakeLists.txt
# [LICENSE] Apache License 2.0 Compliance Embedded.
# ====================================================================
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(pim_hbm_bypass LANGUAGES CXX CUDA)

# [1] C++ 표준 규격 및 호스트 컴파일러 엄격한 정합성 설정
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# [2] 하드웨어 가속 컴파일러(NVCC) 전용 아키텍처 및 고속 연산 최적화 플래그 강제 고정
# 가중치 데이터 로드 캐시 라인 가속을 위해 전역 최적화 레벨 3(-O3) 및 fast_math를 바인딩합니다.
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # 기본 사양으로 Ampere(sm_80) 및 Hopper(sm_90) 아키텍처 바이너리를 2중 빌드하도록 강제합니다.
    set(CMAKE_CUDA_ARCHITECTURES "80;90")
endif()

# 컴파일러가 조건부 이동 명령어(SEL)를 더 유연하게 추출하도록 컴파일 지시문을 추가합니다.
list(APPEND CMAKE_CUDA_FLAGS "-O3" "--use_fast_math" "-Xcompiler" "-fPIC")

# [3] 호스트 환경 내 pybind11 인프라 자동 감지 및 패키지 락
execute_process(
    COMMAND python3 -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(NOT pybind11_DIR)
    message(FATAL_ERROR "[BUILD_FAILURE] 시스템 환경 내 pybind11 패키지를 발견하지 못했습니다. 'pip install pybind11'을 가동하십시오.")
endif()

find_package(pybind11 REQUIRED)

# [4] PIM-HBM 코어 실리콘 링킹 타겟 정의 및 공유 라이브러리(.so) 속성 지정
# 파이썬에서 'import pim_hbm_bridge_core'로 직통 유입되도록 라이브러리 이름을 바인딩합니다.
pybind11_add_module(pim_hbm_bridge_core MODULE pim_hbm_core.cu)

# 컴파일러가 CUDA 런타임 내장 프리미티브(device_launch_parameters.h 등)를 손실 없이 읽도록 바인딩
set_target_properties(pim_hbm_bridge_core PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    PREFIX ""
    OUTPUT_NAME "pim_hbm_bridge_core"
)

# [5] 헤더 포함 경로 명시화 방화벽 가동
target_include_directories(pim_hbm_bridge_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

# [6] 최종 시스템 링킹 마감 리포트 출력
message(STATUS "====================================================================")
message(STATUS "🎯 PIM-HBM HARDWARE CO-DESIGN ENGINE CMAKE BUILD CONFIGURATION")
message(STATUS " - Target CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS " - Pybind11 Directory Found:  ${pybind11_DIR}")
message(STATUS "====================================================================")
