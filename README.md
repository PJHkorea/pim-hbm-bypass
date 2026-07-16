# 🚀 pim-hbm-bypass Blueprint

> **Experimental PIM-HBM Hardware Co-Design Subsystem exploring 0ns Framework Memory View Fusion & Fault Telemetry**

This project is a **hardware-software co-design prototype** engineered to investigate and resolve software abstraction fragmentation barriers within next-generation accelerator infrastructure environments.

By interlocking the low-level physical cache-line alignment mechanisms directly with the upper high-performance framework (JAX/XLA) address interfaces into a unified algebraic pipeline, this subsystem explores the structural viability of stably mitigating static compilation stalls and hardware timing jitters during distributed cluster operations.


---

## ⚡ Key Innovations

1. **0ns Memory Copy Bypass**: Leverages the `__cuda_array_interface__` specification to directly link low-level C++ physical address lines to the JAX tensor bus, structurally eliminating host-device hardware memory copy overhead.
2. **Pure Branchless Loop**: Completely eradicates conditional branches (`if`), instead deploying precise ternary operations and register-resident data reuse patterns to guarantee the compiler forces conditional move primitives (`SEL/PRMT`).
3. **Warp-level Dynamic Bounds**: Intercepts potential out-of-bound memory faults (`SegFault`) inside ragged tail grids by driving a warp-level binary tree reduction firewall via `__shfl_down_sync` to compute active maximum surviving address offsets.
4. **Algebraic Insulation Gate**: Captures hardware fault signals and floating-point divergence anomalies behind the JAX runtime using dedicated `stop_gradient` circuits, isolating backpropagation chain contamination at the physical layer.
5. **Dynamic Hot-Plugging Recovery**: Upon physical HBM bank failure during active cluster runs, this mechanism attempts real-time 64-bit address wire hot-swapping for the corrupted device slot without causing collective communication (NCCL) stalls or graph recompilation.


## 📂 Repository Structure

- **`LICENSE`**: Declares legal safeguards and patent retaliation defense clauses under the Apache License 2.0 specification.
- **`CMakeLists.txt`**: The build orchestrator that automatically tracks system hardware architecture topologies and `pybind11` compilation paths to emit the final shared object (`.so`) libraries.
- **`pim_hbm_core.cu`**: The core branchless mathematical acceleration kernel implementing `alignas(32)` cache-line matching, `__activemask()` dynamic address firewalls, and `__ldg` high-speed read rails (C++/CUDA).
- **`pim_hardware_gate.py`**: The pre-warming and backpropagation chain insulation layer utilizing `ShapeDtypeStruct` virtual abstract tracers to lock XLA compiler machine code while maintaining 0MB of physical VRAM footprint (Python/JAX).
- **`topology_sharding.py`**: The macro-level topology control tower that intercepts per-node VRAM physical address lines to establish zero-copy `NamedSharding` global distributed matrix views (Python/JAX).
- **`hardware_fault_recovery.py`**: The real-time, high-availability hot-plugging swap engine that handles background fault scans within the distributed weight matrices and switches routing to the pre-reserved emergency backup address pool (Python/JAX).
- **`hardware_fault_recovery_distributed.py`**: The collective scanning and recovery engine engineered for large-scale infrastructure, leveraging wire-level NCCL All-Reduce fusion and `np.flatnonzero` vectorized fault extraction (Python/JAX).
- **`llama3_layer_adapter.py`**: A transformer layer adapter plugin that conducts direct 0ns address ingestion matching Llama-3-8B dimensions (4096 / 14336) alongside branchless fault-tolerant forward execution buses (Python/JAX).


---

## 🛠️ Quick Start

Execute the following commands sequentially within a high-performance cluster terminal configured with NVIDIA Ampere (A100) or Hopper (H100/H200) environments to ignite the memory view fusion mode and the hardware fault-tolerant emulation engine.

```bash
# 1. Create and enter an isolated directory dedicated to the build sequence
mkdir build && cd build

# 2. Launch cross-compilation architecture scanning (Automatically tracks pybind11 and CUDA paths)
cmake ..

# 3. Build and compile the hardware machine-code library (Extracts pim_hbm_bridge_core.so)
make -j\$(nproc)

# 4. Migrate the extracted shared library module back to the upper execution directory
cp pim_hbm_bridge_core*.so .. && cd ..

# 5. [STEP A] Execute the single-node JAX pre-warming engine and algebraic insulation guard
python3 pim_hardware_gate.py

# 6. [STEP B] Launch the macro-level distributed sharding topology virtual view fusion matrix
python3 topology_sharding.py

# 7. [STEP C] Run the collective health scan and dynamic hot-plugging recovery via wire-level NCCL All-Reduce
python3 hardware_fault_recovery_distributed.py

# 8. [⚡ STEP D] Instantiate the zero-copy address adapter infrastructure tailored for Llama-3-8B (4096/14336) matrices
python3 llama3_layer_adapter.py
```


---

## 📜 License

This project is distributed under the terms of the **Apache License 2.0**. You are free to modify and distribute the software, provided that the original copyright notice and license disclosure obligations are fully preserved.

---

# 🚀 pim-hbm-bypass Blueprint

> **Experimental PIM-HBM Hardware Co-Design Subsystem exploring 0ns Framework Memory View Fusion & Fault Telemetry**

본 프로젝트는 차세대 가속기 인프라 환경에서 발생할 수 있는 소프트웨어 계층 간 파편화 장벽을 연구하기 위해 설계된 **하드웨어-소프트웨어 공동 설계(Co-design) 프로토타입**입니다. 

저수준의 물리 캐시라인 정렬 매커니즘과 상위 고성능 프레임워크(JAX/XLA) 간의 주소선 인터페이스를 단일 대수 파이프라인으로 연결하여, 분산 클러스터 가동 중 유발되는 정적 컴파일 렉 및 하드웨어 지터를 안정적으로 제어할 수 있는 가능성을 탐구합니다.

---

## ⚡ 핵심 아키텍처 특성 (Key Innovations)

1. **0ns Memory Copy Bypass**: `__cuda_array_interface__` 규격을 활용해 C++ 기계어 주소선을 JAX 텐서 버스에 직결함으로써, 호스트-디바이스 간 물리 복사 오버헤드를 구조적으로 해소합니다.
2. **Pure Branchless Loop**: 조건 분기문(`if`)을 완전히 배제하고 삼항 연산 및 레지스터 상주 데이터 재사용 구조를 정밀 구성하여, 컴파일러 수준의 조건부 이동 명령어(`SEL/PRMT`) 출력을 유도합니다.
3. **Warp-level Dynamic Bounds**: 마지막 그리드 자투리 영역에서 발생할 수 있는 메모리 참조 오류(SegFault)를 방지하기 위해, `__shfl_down_sync` 기반 워프 내 2진 트리 최대 생존 주소 동적 리덕션 방화벽을 구동합니다.
4. **Algebraic Insulation Gate**: JAX 런타임에서 하드웨어 결함 신호 및 수치 발산 오차를 `stop_gradient` 회로로 포획하여, 미분 사슬 오염을 피지컬 레벨에서 격리 및 절연합니다.
5. **Dynamic Hot-Plugging Recovery**: 가속기 클러스터 구동 중 특정 HBM 뱅크에 물리적 결함 발생 시, 글로벌 통신(NCCL) 중단 및 그래프 재컴파일 없이 오직 불량 장치의 64비트 주소선만 실시간으로 우회 스와핑(Hot-Swapping)하는 메커니즘을 시도합니다.

## 📂 파일 토폴로지 (Repository Structure)

- **`LICENSE`**: Apache License 2.0 의거 법적 방화벽 및 특허 보호 조항 명시
- **`CMakeLists.txt`**: 시스템 가속기 아키텍처 환경과 `pybind11` 컴파일 패스를 자동 추적하여 공유 라이브러리(`.so`)를 사출하는 빌드 오케스트레이터
- **`pim_hbm_core.cu`**: `alignas(32)` 캐시라인 일치 레이아웃, `__activemask()` 동적 주소 방화벽 및 `__ldg` 가속 레일이 주입된 무분기 수학 가속 커널 코어 (C++/CUDA)
- **`pim_hardware_gate.py`**: `ShapeDtypeStruct` 가상 추상화 트레이서를 활용해 실재 VRAM 점유 0MB 상태로 XLA 컴파일러 기계어를 고정하는 예열 및 미분 사슬 절연 레이어 (Python/JAX)
- **`topology_sharding.py`**: 대규모 클러스터 노드별 VRAM 물리 주소선을 가로채어 제로카피 `NamedSharding` 글로벌 분산 매트릭스 뷰를 수립하는 거시적 토폴로지 관제탑 (Python/JAX)
- **`hardware_fault_recovery.py`**: 분산 가중치 행렬 내 불량 뱅크 백그라운드 스캔 및 비상 백업 풀 주소선을 활용한 실시간 무중단 핫플러깅 스와프 엔진 (Python/JAX)
- **`hardware_fault_recovery_distributed.py`**: 초대형 인프라를 위한 NCCL All-Reduce 와이어 레벨 융합 집산(Collective) 스캔 및 `np.flatnonzero` 벡터화 결함 적출 복구 엔진 (Python/JAX)
- **`llama3_layer_adapter.py`**: Llama-3-8B 고유 차원(4096 / 14336) 직통 0ns 주소 인입 및 무분기 결함 허용 순방향 훈련/추론 버스 어댑터 플러그인 (Python/JAX)

---

## 🛠️ 고속 구동 및 빌드 지침 (Quick Start)

NVIDIA Ampere(A100) 또는 Hopper(H100/H200) 환경이 구축된 고성능 클러스터 터미널에서 다음 명령을 순차 가동하여 바이패스 모드 및 하드웨어 결함 허용(Fault-Tolerant) 엔진 에뮬레이션을 기폭할 수 있습니다.

```bash
# 1. 빌드 전용 격리 공간 생성 및 진입
mkdir build && cd build

# 2. 크로스 컴파일 아키텍처 스캔 가동 (pybind11 및 CUDA 컴파일 패스 자동 추적)
cmake ..

# 3. 하드웨어 기계어 라이브러리 컴파일 빌드 (pim_hbm_bridge_core.so 추출)
make -j\$(nproc)

# 4. 상위 실행 디렉토리로 추출된 모듈 복사 이관
cp pim_hbm_bridge_core*.so .. && cd ..

# 5. [STEP A] 단일 노드 전용 JAX 예열 엔진 및 대수적 오차 절연 가드 가동
python3 pim_hardware_gate.py

# 6. [STEP B] Multi-GPU 거시 분산 샤딩 토폴로지 가상 융합 텐서 가동 
python3 topology_sharding.py

# 7. [STEP C] 초대형 인프라용 NCCL All-Reduce 융합 분산 집산 헬스 스캔 및 핫플러깅 복구 가동
python3 hardware_fault_recovery_distributed.py

# 8. [⚡ STEP D] 실전 Llama-3-8B Transformer 4096/14336 매트릭스 0ns 주소 제로카피 어댑터 인프라 가동
python3 llama3_layer_adapter.py
```

---

## 📜 라이선스 (License)

본 프로젝트는 **Apache License 2.0** 의거하여 배포됩니다. 자유로운 수정 및 배포가 가능하나 저작권 및 라이선스 고지 의무가 수반됩니다.

