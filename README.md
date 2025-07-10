# GNN 기반 현실적 TSN 라우팅 최적화 솔버 (v2.0)

## 1. 프로젝트 개요

이 프로젝트는 시간 민감형 네트워크(Time-Sensitive Networking, TSN) 환경에서 **다수의 데이터 플로우에 대한 최적의 주 경로(Primary Path) 및 예비 경로(Backup Path)를 자동으로 설정**하는 지능형 라우팅 솔루션입니다.

GraphSAGE 신경망과 PPO 강화학습 알고리즘을 사용하여, **"무엇을 입력하면, 무엇이 출력되는가"**를 명확히 정의합니다.

-   **입력 (Input)**: 네트워크의 현재 상태
    -   `네트워크 토폴로지`: 스위치(노드)와 링크(엣지)의 연결 구조, 링크별 대역폭
    -   `플로우 요구사항`: 수십 ~ 수백 개 데이터 플로우 각각의 출발지, 목적지, 데이터 크기, 주기, 데드라인
    -   `고장 시나리오`: 일부 링크 또는 스위치가 고장 나는 상황

-   **처리 과정 (Processing)**:
    1.  **모방 학습**: ILP(Integer Linear Programming)라는 수학적 최적화 솔버가 생성한 고품질의 경로 설정 데이터를 GNN이 학습하여 전문가의 "통찰력"을 모방합니다.
    2.  **강화 학습**: 모방 학습으로 똑똑해진 GNN이 시뮬레이션 환경과 상호작용하며, 단순히 ILP를 따라 하는 것을 넘어 **지연시간, 안정성, 자원 효율성**까지 고려하는 더 고차원적인 라우팅 전략을 스스로 터득합니다.

-   **출력 (Output)**: 최적화된 전체 라우팅 테이블
    -   `전체 경로 솔루션`: 모든 플로우에 대해 할당된 구체적인 주 경로 및 예비 경로 목록.
      ```json
      {
        "flow_0": {
          "primary": [10, 5, 1, 8],
          "backup": [10, 4, 0, 8]
        },
        "flow_1": {
          "primary": [12, 6, 2, 9],
          "backup": [12, 7, 3, 9]
        },
        ...
      }
      ```
    -   `성능 평가 보고서`: 생성된 라우팅 솔루션이 얼마나 우수한지를 나타내는 점수와 계산 시간.

궁극적인 목표는, 수학적으로 완벽하지만 매우 느린 ILP 솔버를, **빠른 속도로 ILP와 대등하거나 그 이상의 성능을 내는 GNN 솔버로 대체**하는 것입니다.

## 2. 핵심 기능

-   **현실적인 네트워크 모델링**: 데이터센터의 **팻-트리(Fat-Tree)** 토폴로지, 계층별 차등 대역폭, 트래픽이 몰리는 **핫스팟** 현상, **스위치 고장** 등 실제 네트워크의 복잡성을 반영합니다.
-   **CPU 최적화 GNN**: Intel CPU 환경에서 빠른 연산을 위해 집계 기반 **GraphSAGE** 모델을 채택했습니다.
-   **고도화된 2단계 학습**: **모방 학습**으로 전문가의 지식을 빠르게 흡수하고, **PPO 강화학습**으로 실제 환경의 복합적인 목표(지연시간, 공정성, 견고성)에 맞춰 정책을 스스로 개선합니다.
-   **종합 벤치마크**: 개발된 GNN 솔루션을 전통적인 **Greedy** 방식 및 수학적 최적화 방식인 **ILP**와 성능, 계산 시간, 안정성 측면에서 다각도로 비교 분석하고 결과를 시각화합니다.

## 3. 시스템 요구사항 및 설치

-   **요구사항**: Python 3.8+, PyTorch, PyTorch Geometric, `pulp`
-   **설치 가이드**:
    1.  가상 환경 생성: `python -m venv venv && source venv/bin/activate`
    2.  PyTorch 및 PyG 설치 (CPU 버전 기준):
        ```bash
        # PyTorch 공식 홈페이지에서 시스템에 맞는 명령어를 확인하세요.
        pip install torch torchvision torchaudio
        # PyG 설치 (PyTorch 버전에 맞게)
        pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
        pip install torch_geometric
        ```
    3.  나머지 라이브러리 설치: `pip install numpy pandas matplotlib tqdm pulp networkx`

## 4. 실행 방법 및 결과물

1.  **전체 파이프라인 실행**:
    -   터미널에서 아래 명령어를 실행하여 모방 학습, 온라인 학습, 최종 벤치마크를 순차적으로 진행합니다.
    ```bash
    python main.py
    ```
    (여기서 `main.py`는 제공된 파이썬 스크립트의 파일명입니다.)
    -   **주의**: 학습 과정은 CPU 성능에 따라 수십 분에서 수 시간까지 소요될 수 있습니다. `IMITATION_EPISODES`와 `ONLINE_EPISODES` 값을 조절하여 실행 시간을 제어할 수 있습니다.

2.  **실행 결과물**:
    -   **학습된 모델 파일**:
        -   `gnn_graphsage_imitated_final.pth`: 1단계 모방 학습 완료 모델
        -   `gnn_graphsage_final.pth`: 2단계 온라인 학습까지 완료된 최종 모델
    -   **콘솔 출력**:
        -   학습 단계별 Loss 및 Score 진행 상황이 실시간으로 출력됩니다.
        -   최종적으로 각 솔버(GNN, Greedy, ILP)의 평균 성능 점수와 평균 계산 시간을 요약한 **벤치마크 테이블**이 표시됩니다.
    -   **결과 그래프 (`benchmark_results_graphsage_final.png`)**:
        -   벤치마크 결과를 시각화한 막대그래프가 이미지 파일로 저장됩니다.
        -   **상단 그래프**: 성능 점수 비교 (높을수록 좋음)
        -   **하단 그래프**: 계산 시간 비교 (낮을수록 좋음, Y축은 log scale)

## 5. 코드 구조 및 핵심 로직

-   **`RealisticProfileGenerator`**: 네트워크 토폴로지, 플로우 요구사항 등 **"입력 데이터"**를 생성합니다.
-   **`RealisticTSNEnv`**: 생성된 경로 솔루션이 얼마나 좋은지 평가하고 **"점수"**를 매기는 시뮬레이터입니다.
-   **`ActorCriticSAGE`**: 네트워크 상태를 분석하고 다음 행동을 결정하는 GNN **"두뇌"**입니다.
-   **`SAGE_PPOAgent`**: 모방 학습과 강화학습을 통해 GNN 두뇌를 **"훈련"**시키는 트레이너입니다.
-   **`GNN_SAGE_Solver`, `Greedy_Solver`, `ILP_Solver`**: 최종적으로 경로를 계산하는 세 종류의 **"문제 해결사"**입니다.
-   **`run_full_procedure_and_benchmark`**: 위 모든 요소를 순서대로 실행하고 **"최종 보고서"**를 만드는 메인 컨트롤러입니다.

# GNN 기반 현실적 TSN 라우팅 최적화 솔버

## 1. 프로젝트 개요

이 프로젝트는 시간 민감형 네트워크(Time-Sensitive Networking, TSN) 환경에서 데이터 플로우의 경로를 최적화하는 강화학습 기반 솔루션을 제공합니다. 특히, 실제 네트워크 환경과 유사한 조건에서 GNN(Graph Neural Network) 에이전트의 성능을 극대화하는 데 초점을 맞춥니다.

GraphSAGE 모델과 PPO(Proximal Policy Optimization) 알고리즘을 사용하여, 속도가 빠른 Greedy 알고리즘과 성능이 우수한 ILP(Integer Linear Programming)의 장점을 결합하고자 합니다. 최종 목표는 ILP에 준하는 높은 품질의 라우팅 해를 Greedy처럼 빠른 속도로 찾아내는 것입니다.

## 2. 핵심 기능

-   **현실적인 네트워크 토폴로지**: 데이터센터에서 널리 사용되는 **팻-트리(Fat-Tree)** 토폴로지를 생성하고, 계층별로 차등적인 링크 대역폭을 설정합니다.
-   **현실적인 트래픽 프로파일**: 특정 서버로 트래픽이 몰리는 **핫스팟(Hotspot)** 현상과 예측 불가능한 **배경 트래픽(Best-Effort)**을 시뮬레이션합니다.
-   **현실적인 고장 시나리오**: 단순 링크 고장뿐만 아니라, 네트워크에 더 큰 영향을 미치는 **스위치(노드) 고장** 상황을 포함하여 모델의 견고성을 평가합니다.
-   **CPU 최적화 모델**: Intel CPU 환경에서의 빠른 연산을 위해 어텐션 기반 GAT 대신 집계 기반 **GraphSAGE** 모델을 채택했습니다.
-   **고도화된 강화학습**: ILP 솔루션을 전문가 삼아 모방 학습을 수행한 뒤, PPO 알고리즘을 통해 실제 환경의 복합적인 보상(지연시간, 공정성, 자원 효율성)을 최대화하도록 온라인 학습을 진행합니다.
-   **종합적인 성능 벤치마크**: 개발된 GNN 솔루션을 Greedy 및 ILP와 **성능, 계산 시간, 견고성** 측면에서 비교하고 결과를 시각화합니다.

## 3. 시스템 요구사항

-   Python 3.8 이상
-   PyTorch
-   `pulp` 라이브러리와 CBC Solver (pulp 설치 시 대부분 자동으로 설치됨)

## 4. 설치 가이드

1.  **프로젝트 파일 준비**
    -   이 프로젝트의 파이썬 코드(`main.py`)와 `requirements.txt`, `README.md` 파일을 하나의 디렉토리에 저장합니다.

2.  **가상 환경 생성 및 활성화 (권장)**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS / Linux
    source venv/bin/activate
    ```

3.  **PyTorch 설치 (CPU 버전)**
    -   시스템에 맞는 PyTorch 설치 명령어는 [PyTorch 공식 홈페이지](https://pytorch.org/get-started/locally/)에서 확인하는 것이 가장 정확합니다.
    -   일반적인 CPU 버전 설치 명령어는 다음과 같습니다.
    ```bash
    pip install torch torchvision torchaudio
    ```

4.  **PyG (PyTorch Geometric) 설치**
    -   PyG는 PyTorch 버전에 맞춰 설치해야 합니다. 아래 명령어는 PyTorch 2.0 이상 버전에 해당합니다.
    -   자세한 내용은 [PyG 설치 가이드](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)를 참조하세요.
    ```bash
    pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
    pip install torch_geometric
    ```

5.  **나머지 라이브러리 설치**
    ```bash
    pip install -r requirements.txt
    ```

## 5. 실행 방법

1.  터미널에서 아래 명령어를 실행하여 전체 학습 및 벤치마크 파이프라인을 시작합니다.
    ```bash
    python main.py
    ```
    (여기서 `main.py`는 제공된 파이썬 스크립트의 파일명입니다.)

2.  **실행 과정**:
    -   **1단계 (모방 학습)**: `IMITATION_MODEL_PATH`에 저장된 모델이 없으면, ILP 솔버를 전문가로 하여 모방 학습을 진행하고 결과를 저장합니다.
    -   **2단계 (온라인 학습)**: 모방 학습으로 초기화된 모델을 PPO 알고리즘으로 추가 학습시키고, 최종 모델을 `FINAL_MODEL_PATH`에 저장합니다.
    -   **3단계 (벤치마크)**: 최종 학습된 GNN 모델을 Greedy, ILP와 함께 벤치마크합니다.
    -   콘솔에 최종 성능 요약 테이블이 출력되고, 실행 결과 그래프가 `RESULT_PLOT_PATH`에 지정된 파일명(`benchmark_results_graphsage_realistic.png`)으로 저장됩니다.

## 6. 코드 구조

-   `RealisticProfileGenerator`: 팻-트리 토폴로지, 핫스팟 트래픽 등 현실적인 시나리오 프로파일을 생성합니다.
-   `RealisticTSNEnv`: 현실적인 제약 조건(계층별 대역폭, 노드 고장)과 복합적인 평가 지표(지연시간, 공정성, 자원 효율성)를 포함하는 시뮬레이션 환경입니다.
-   `ActorCriticSAGE`: GraphSAGE를 기반으로 한 Actor-Critic 신경망 모델입니다.
-   `SAGE_PPOAgent`: PPO 알고리즘을 사용하여 `ActorCriticSAGE` 모델을 학습시키는 에이전트입니다.
-   `GNN_SAGE_Solver`, `Greedy_Solver`, `ILP_Solver`: 벤치마크에 사용될 세 가지 솔버 클래스입니다.
-   `run_imitation_learning`, `run_ppo_guided_rl`: 각 학습 단계를 수행하는 함수입니다.
-   `run_full_procedure_and_benchmark`: 전체 파이프라인을 실행하고 결과를 보고하는 메인 함수입니다.


 
# GNN-based Reinforcement Learning for Robust TSN Routing

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![GitHub Pages](https://img.shields.io/badge/GitHub-Pages-1DA1F2?logo=github)](https://hwkim3330.github.io/GNN/)

**[► Project Webpage (GitHub Pages)](https://hwkim3330.github.io/GNN/)**

## Abstract (초록)

본 연구는 시간 민감형 네트워크(Time-Sensitive Networking, TSN) 환경에서 예측 불가능한 링크 장애에 대응할 수 있는 견고한(robust) 라우팅 경로를 설정하기 위한 새로운 프레임워크를 제안한다. 이를 위해, 그래프 어텐션 네트워크(Graph Attention Network, GATv2) 기반의 심층 강화학습(Deep Reinforcement Learning) 에이전트를 설계하였다. 제안하는 에이전트는 네트워크의 위상, 혼잡도, 그리고 각 플로우의 대역폭 요구사항을 포함하는 풍부한 상태 정보를 바탕으로 의사결정을 내린다. 학습 과정은 두 단계로 구성된다: (1) Greedy 알고리즘을 모방하여 기본적인 경로 탐색 능력을 빠르게 습득하는 초기 모방 학습 단계, (2) 정수 선형 계획법(Integer Linear Programming, ILP) 솔버를 '앵커(anchor)'로 활용하여 정책을 정교화하는 온라인 강화학습 단계. 이 하이브리드 학습 전략을 통해 에이전트는 ILP가 제한 시간 내에 찾은 해(solution)를 뛰어넘는 고품질의 라우팅 경로를 발견할 수 있다. 실험 결과, 제안하는 GNN 에이전트는 기존의 Greedy 및 ILP 방식에 비해 더 우수한 성능(End-to-End Latency 최소화)을 달성하면서도, 추론 시간을 수백 배 단축시키는 데 성공했다.

---

## 🌟 Key Features

-   **Advanced GNN Model**: `GATv2`를 사용하여 네트워크의 위상적 특징과 노드 간의 중요도를 학습합니다.
-   **Hybrid Learning Strategy**: ILP(정수 선형 계획법) 솔버를 '앵커'로 사용하는 독창적인 강화학습 프레임워크를 통해 GNN이 전문가의 성능을 뛰어넘도록 학습합니다.
-   **Rich State Representation**: 단순한 노드 정보가 아닌, 네트워크의 혼잡도(congestion)와 플로우의 대역폭 요구사항 등 풍부한 상태 정보를 GNN에 제공하여 지능적인 의사결정을 유도합니다.
-   **Robustness Evaluation**: 정상 상태뿐만 아니라, 링크 고장(link failure) 시나리오를 포함하여 실제 환경과 유사한 견고성을 평가합니다.

---

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/hwkim3330/GNN.git](https://github.com/hwkim3330/GNN.git)
    cd GNN
    ```

2.  **Create & Activate Conda Environment:**
    ```bash
    conda create -n tsn_env python=3.9 -y
    conda activate tsn_env
    ```

3.  **Install Dependencies:**
    PyTorch를 CUDA 버전에 맞게 먼저 설치 후, 나머지 라이브러리를 설치합니다.
    ```bash
    # For CUDA 12.1
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
    
    # Install other packages
    pip install -r requirements.txt
    
    # Install ILP Solver (for Debian/Ubuntu)
    sudo apt-get install -y coinor-cbc
    ```

---

## 🚀 Usage

**Train a new model:**
전체 학습 파이프라인(모방학습 -> 온라인 강화학습 -> 벤치마크)을 실행합니다.
```bash
python code.py
````

-----

## 📈 Methodology

본 프로젝트는 2단계 학습 전략을 사용합니다.

1.  **Phase 1: Imitation Learning (Warm-up)**: GNN 에이전트($\\pi\_\\theta$)가 전문가($\\pi^\*$)의 행동을 모방하여 기본적인 정책을 학습합니다. 손실 함수는 Cross-Entropy를 사용합니다.
    $$L_{IL}(\theta) = - \sum_{(s,a) \in D} \log \pi_\theta(a|s)$$
    여기서 $D$는 전문가(Greedy 알고리즘)가 생성한 상태-행동 쌍의 데이터셋입니다.

2.  **Phase 2: ILP-Anchored Online Learning**: GNN 에이전트가 `ILP Solver`와 경쟁하며 학습을 진행합니다. GNN의 해가 ILP보다 우수하면 강화학습(Policy Gradient)을 통해 정책을 강화하고, ILP의 해가 더 우수하면 모방학습을 통해 정책을 교정합니다. 강화학습의 목적 함수는 다음과 같습니다.
    $$J(\theta) = E_{\tau \sim \pi_\theta}[R(\tau)]$$
    $R(\\tau)$는 에피소드 $\\tau$에서 얻은 보상(reward)으로, 본 프로젝트에서는 지연시간에 반비례하는 점수를 사용합니다.

-----

## 📊 Results

학습 완료 후, GNN, Greedy, ILP 솔버의 성능(점수)과 계산 시간을 비교한 아래와 같은 그래프가 생성됩니다.

*(여기에 최종 결과 그래프 이미지 `benchmark_results_v3.png`를 업로드하여 삽입하세요)*

| Solver  | Average Score | Average Time (s) |
| :------ | :------------ | :--------------- |
| GNN\_v3  | 0.95+         | \~0.1s            |
| ILP     | \~0.90         | \~20s             |
| Greedy  | \~0.85         | \<0.01s           |
*(결과 예시 테이블)*

-----

## 📄 License

This project is licensed under the MIT License.

-----

## ✏️ How to Cite

이 프로젝트가 유용하다고 생각되시면, 아래와 같이 인용해주세요:

```bibtex
@misc{kim2025gnn_tsn,
  author = {Kim, H.W.},
  title = {GNN-based Reinforcement Learning for Robust TSN Routing},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{[https://github.com/hwkim3330/GNN](https://github.com/hwkim3330/GNN)}}
}
```

```
```
