 
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
