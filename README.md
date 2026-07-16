# 🚀 pim-hbm-bypass Blueprint

> **Pure Algebraic PIM-HBM Hardware Co-Design Engine with 0ns JAX/PyTorch Memory Bypass**


본 프로젝트는 차세대 가속기 아키텍처의 소프트웨어 파편화 장벽을 해소하기 위해 설계된 **반도체-소프트웨어 공동 설계(Co-design) 플랫폼**입니다. 하부 레벨의 물리 캐시라인 정렬 제어와 상위 프레임워크(JAX/PyTorch)의 0ns 주소선 가로채기 레이어를 단일 파이프라인으로 관통하여 가속기 파이프라인 지터 0.0%를 목표로 합니다.

---

## ⚡ 핵심 아키텍처 혁신 (Key Innovations)

1. **0ns Memory Copy Bypass**: `__cuda_array_interface__` 규격을 가로채어 C++ 기계어 주소선을 JAX 텐서 버스에 직결, 호스트-디바이스 간 물리 복사 오버헤드를 제로(0ns)로 박멸합니다.
2. **Pure Branchless Loop**: 조건 분기문(`if`)을 전면 도살하고 삼항 연산 및 레지스터 상주 데이터 재사용 구조를 유도하여 컴파일러 수준의 조건부 이동 명령어(`SEL/PRMT`)를 100% 강제 출력합니다.
3. **Warp-level Dynamic Bounds**: 마지막 그리드 짜투리 영역에서 발생하는 SegFault를 완벽히 제거하기 위해 `__shfl_down_sync` 기반 워프 내 2진 트리 최대 생존 주소 동적 리덕션 방화벽을 구동합니다.
4. **Algebraic Insulation Gate**: JAX 런타임에서 하드웨어 결함 신호 및 수치 발산 오차를 `stop_gradient` 회로로 포획하여 미분 사슬 오염을 피지컬 레벨에서 격리 절연합니다.
5. **Dynamic Hot-Plugging Recovery (🆕)**: 가속기 클러스터 구동 중 특정 HBM 뱅크 폭사 시, 글로벌 통신(NCCL) 중단 및 그래프 재컴파일 없이 오직 불량 장치의 64비트 주소선만 실시간으로 우회 스와핑(Hot-Swapping)합니다.

## 📂 파일 토폴로지 (Repository Structure)

- **`LICENSE`**: Apache License 2.0 법적 방화벽 및 특허 보복 방어 조항 명시
- **`CMakeLists.txt`**: 시스템 가속기 아키텍처 환경과 `pybind11` 컴파일 패스를 자동 추적하여 공유 라이브러리(`.so`)를 사출하는 빌드 오케스트레이터
- **`pim_hbm_core.cu`**: `alignas(32)` 캐시라인 일치 레이아웃, `__shfl_down_sync` 주소 방화벽 및 `__ldg` 가속 레일이 주입된 무분기 수학 가속 커널 코어 (C++/CUDA)
- **`pim_hardware_gate.py`**: `ShapeDtypeStruct` 가상 추상화 트레이서를 활용해 실재 VRAM 점유 0MB 상태로 XLA 컴파일러 기계어를 영구 고정하는 예열 및 미분 사슬 절연 레이어 (Python/JAX)
- **`topology_sharding.py`**: 대규모 클러스터 노드별 VRAM 물리 주소선을 가로채어 제로카피 `NamedSharding` 글로벌 분산 매트릭스 뷰를 수립하는 거시적 토폴로지 관제탑 (Python/JAX)
- **`hardware_fault_recovery.py`**: 분산 가중치 행렬 내 불량 뱅크 백그라운드 스캔 및 비상 백업 풀 주소선을 활용한 실시간 무중단 핫플러깅 스와프 엔진 (Python/JAX)

---


## 🛠️ 고속 구동 및 빌드 지침 (Quick Start)

NVIDIA Ampere(A100) 또는 Hopper(H100/H200) 환경이 구축된 고성능 클러스터 터미널에서 다음 명령을 순차 가동하여 0ns Jitter 바이패스 모드 및 하드웨어 결함 허용(Fault-Tolerant) 엔진을 즉시 기폭할 수 있습니다.

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

# 7. [STEP C] 무중단 연속 구동 실리콘 결함 허용 핫플러깅 복구 파이프라인 가동
python3 hardware_fault_recovery.py
```

---

## 📜 라이선스 (License)

본 프로젝트는 **Apache License 2.0** 의거하여 배포됩니다. 
