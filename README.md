# 🚀 pim-hbm-bypass Blueprint

> **Pure Algebraic PIM-HBM Hardware Co-Design Engine with 0ns JAX/PyTorch Memory Bypass**


본 프로젝트는 차세대 가속기 아키텍처의 소프트웨어 파편화 장벽을 해소하기 위해 설계된 **반도체-소프트웨어 공동 설계(Co-design) 플랫폼**입니다. 하부 레벨의 물리 캐시라인 정렬 제어와 상위 프레임워크(JAX/PyTorch)의 0ns 주소선 가로채기 레이어를 단일 파이프라인으로 관통하여 가속기 파이프라인 지터 0.0%를 목표로 합니다.

---

## ⚡ 핵심 아키텍처 혁신 (Key Innovations)

1. **0ns Memory Copy Bypass**: `__cuda_array_interface__` 규격을 가로채어 C++ 기계어 주소선을 JAX 텐서 버스에 직결, 호스트-디바이스 간 물리 복사 오버헤드를 물리적으로 제거합니다.
2. **Pure Branchless Loop**: 조건 분기문(`if`)을 전면 도살하고 삼항 연산 및 레지스터 상주 데이터 재사용 구조를 유도하여 컴파일러 수준의 조건부 이동 명령어(`SEL/PRMT`)를 강제합니다.
3. **Warp-level Dynamic Bounds**: 마지막 그리드 짜투리 영역에서 발생하는 SegFault를 방어하기 위해 `__shfl_down_sync` 기반 워프 내 2진 트리 최대 주소 리덕션 방화벽을 구동합니다.
4. **Algebraic Insulation Gate**: JAX 런타임에서 하드웨어 결함 신호 및 수치 발산 오차를 `stop_gradient` 회로로 포획하여 미분 사슬 오염을 절연합니다.

---

## 📂 파일 토폴로지 (Repository Structure)

- `LICENSE`: Apache License 2.0 
- `CMakeLists.txt`: `pybind11` 및 `nvcc` 아키텍처 타겟 자동 추적 빌드 오케스트레이터
- `pim_hbm_core.cu`: 32바이트 정렬 실리콘 레이아웃 및 무분기 수학 가속 커널 코어 (C++/CUDA)
- `pim_hardware_gate.py`: OOM 방지 가상 트레이서 기반 XLA 컴파일러 예열 및 절연 레이어 (Python/JAX)

---

## 🛠️ 고속 구동 및 빌드 지침 (Quick Start)

NVIDIA Ampere(A100) 또는 Hopper(H100) 환경이 구축된 터미널에서 다음 명령을 순차 가동하여 0ns Jitter 바이패스 모드를 즉시 기폭할 수 있습니다.

\`\`\`bash
# 1. 빌드 전용 격리 공간 생성 및 진입
mkdir build && cd build

# 2. 크로스 컴파일 아키텍처 스캔 가동 (pybind11 및 CUDA 컴파일 패스 자동 추적)
cmake ..

# 3. 하드웨어 기계어 라이브러리 컴파일 빌드 (pim_hbm_bridge_core.so 추출)
make -j$(nproc)

# 4. 상위 실행 디렉토리로 추출된 모듈 복사 이관
cp pim_hbm_bridge_core*.so .. && cd ..

# 5. JAX 예열 엔진 구동 및 0ns Jitter 방화벽 기폭
python3 pim_hardware_gate.py
\`\`\`

---

## 📜 라이선스 (License)

본 프로젝트는 **Apache License 2.0** 의거하여 배포됩니다. 
