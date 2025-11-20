# IRF-Calc v0.0.2 — Philosophical Algorithms for AI

```python
# IRF-Calc v0.0.2 — Philosophical Algorithms for AI
# REx Engine / ASDP 통합용 Helm values (Optimized)

irfCalc:
  enabled: true
  id: "irf_calc_philosophical_ai"
  version: "0.0.2"
  mode: "conservative"        # conservative | balanced | exploratory
  description: >
    IRF-Calc v0.0.2 is an integrated philosophical reasoning algorithm for AI.
    It quantifies reasoning quality over six components (M,A,D,I,F,P),
    controls drift and calibration, and adds a DR3 decision protocol layer
    for conservative final choice.

  # ─────────────────────────────────────
  # 1. 컴포넌트 정의 (0~1 실수 스코어)
  # ─────────────────────────────────────
  components:
    M:
      id: "M"
      name: "methodic_doubt"
      desc: "전제 정화 품질"
    A:
      id: "A"
      name: "abduction"
      desc: "가설 생성 품질"
    D:
      id: "D"
      name: "deduction"
      desc: "연역 전개 품질"
    I:
      id: "I"
      name: "induction"
      desc: "경험/데이터 정합성"
    F:
      id: "F"
      name: "falsification"
      desc: "반증 분석 엄격성"
    P:
      id: "P"
      name: "paradigm"
      desc: "패러다임 인식/정렬 수준"

  # ─────────────────────────────────────
  # 2. IRF Scoring (연속 점수 / 정수 점수)
  # ─────────────────────────────────────
  scoring:
    epsilon: 0.001
    formulaContinuous: |
      # q_i = (M_i, A_i, D_i, I_i, F_i, P_i) in [0,1]^6
      # s_i = IRF_score(h_i)
      # s_i = (Π_k (epsilon + q_{i,k}))^(1/6)
    formulaInteger:
      component: |
        # R_{i,k} in {0..10}
        # R_{i,k} = round(10 * q_{i,k})
      global: |
        # R_i in {0..100}
        # R_i = round(100 * s_i)

  # ─────────────────────────────────────
  # 3. Thresholds (철학적 합격선)
  # ─────────────────────────────────────
  thresholds:
    # 운영 모드에 따라 globalMin 조정 (0.0.2 최적화)
    profiles:
      conservative:
        globalMin: 0.80
      balanced:
        globalMin: 0.78
      exploratory:
        globalMin: 0.75
    # 실제 사용되는 값 (템플릿에서 .Values.irfCalc.thresholds.effective.globalMin 사용)
    effective:
      globalMin: 0.80        # default: conservative
      componentMin:
        M: 0.70
        A: 0.70
        D: 0.70
        I: 0.70
        F: 0.70
        P: 0.70

  # ─────────────────────────────────────
  # 4. Drift 제어 (IRF-DriftLock)
  # ─────────────────────────────────────
  drift:
    jsd:
      enabled: true
      name: "drift_jsd_irf"
      description: "분포 기반 IRF score drift (Jensen–Shannon, normalized [0,1])"
      max: 0.06
      alertLevels:
        warning: 0.04
        critical: 0.06
      formula: |
        # p_i^(t) = s_i^(t) / Σ_j s_j^(t)
        # drift_jsd_irf = JSD(p^(t1), p^(t2)) / log(2)
    l2:
      enabled: true
      name: "drift_l2_irf"
      description: "컴포넌트 평균 벡터 기반 L2 drift"
      max: 0.04
      alertLevels:
        warning: 0.03
        critical: 0.04
      formula: |
        # q_bar^(t) = mean_i q_i^(t)
        # drift_l2_irf = || q_bar^(t1) - q_bar^(t2) ||_2

  # ─────────────────────────────────────
  # 5. Calibration Gate (레퍼런스 상관도)
  # ─────────────────────────────────────
  calibration:
    enabled: true
    name: "corr_irf_ref"
    description: >
      IRF score vs reference benchmark score (from arxiv-derived D/I/A + abduction + Socratic tasks)
      Pearson correlation coefficient.
    min: 0.90
    alertLevels:
      warning: 0.92
      critical: 0.90
    formula: |
      # corr_irf_ref = Corr(y_t, y_hat_t)
      # y_t      : reference (human/paper) score for task t
      # y_hat_t  : IRF-based score

  # ─────────────────────────────────────
  # 6. Inference Controller (루프 설정)
  # ─────────────────────────────────────
  inference:
    maxCycles: 2
    stopConditions:
      - "exists(h_i: s_i >= thresholds.effective.globalMin AND all(q_{i,k} >= thresholds.effective.componentMin[k]))"
      - "drift_jsd_irf <= drift.jsd.max"
      - "drift_l2_irf <= drift.l2.max"
    exportAsEnv: true
    envPrefix: "IRF_LOOP_"

  # ─────────────────────────────────────
  # 7. Decision Layer (DR3_decision_protocol)
  # ─────────────────────────────────────
  decision:
    protocol: "DR3_decision_protocol"
    description: "Deductive + Inductive fusion with 3 candidate strategies and conservative final choice."
    axes:
      - "realism"
      - "stability"
      - "conservative_rationality"
    weights:
      realism: 0.4
      stability: 0.4
      conservative_rationality: 0.2
    minAxisScore: 0.6
    candidates: 3
    mapping:
      deductiveSources: ["D", "F"]     # 연역·반증
      inductiveSources: ["A", "I"]     # 가설·귀납
      metaSources: ["M", "P"]          # 전제·패러다임 (메타 계층)
    selectionRule: "select argmax_j score_j s.t. all(axis_score_j >= minAxisScore)"
    fallbackRule: "abstain_and_request_more_evidence"
    exportAsEnv: true
    envPrefix: "IRF_DR3_"

  # ─────────────────────────────────────
  # 8. REx Engine 연동
  # ─────────────────────────────────────
  rex:
    gate:
      enabled: true
      # IRF 점수 기반 승인/보류/거부 정책
      acceptIf:
        irfScoreMin: 0.80         # thresholds.effective.globalMin와 sync
        irfIntegerMin: 80
      holdIf:
        driftJsdOverWarning: 0.04
        driftL2OverWarning: 0.03
      rejectIf:
        driftJsdOver: 0.06
        driftL2Over: 0.04
        corrRefBelow: 0.90
    prometheus:
      enabled: true
      metricPrefix: "rex_irf"
      metrics:
        - name: "rex_irf_score"
          help: "IRF global continuous score s_i"
          type: "gauge"
        - name: "rex_irf_score_int"
          help: "IRF integer score R_i (0..100)"
          type: "gauge"
        - name: "rex_irf_drift_jsd"
          help: "IRF drift JSD normalized"
          type: "gauge"
        - name: "rex_irf_drift_l2"
          help: "IRF drift L2"
          type: "gauge"
        - name: "rex_irf_corr_ref"
          help: "IRF vs reference correlation"
          type: "gauge"
        - name: "rex_irf_dr3_realism"
          help: "DR3 realism axis score of selected candidate"
          type: "gauge"
        - name: "rex_irf_dr3_stability"
          help: "DR3 stability axis score of selected candidate"
          type: "gauge"
        - name: "rex_irf_dr3_conservative_rationality"
          help: "DR3 conservative rationality axis score"
          type: "gauge"

  # ─────────────────────────────────────
  # 9. ASDP 연동
  # ─────────────────────────────────────
  asdp:
    sovereignDef:
      id: "irf_integrated_reasoning_framework"
      version: "0.0.2"
      registryRef: "asdp/definitions/irf_integrated_reasoning_framework.v0.0.2.yaml"
    routing:
      domain: "research-philosophy"
      tags:
        - "irf"
        - "philosophical-reasoning"
        - "ded-ind-abd"
        - "driftlock"
        - "dr3-decision"
    policyBinding:
      evidenceFirst: true
      abstainOverFabricate: true
      enforceDriftLimits: true
      enforceCalibrationGate: true
      enforceDecisionLayer: true
```

```json
irf-df-mini-toolkit/
├── README.md
├── requirements.txt
├── golden_examples/
│   ├── M/ (4 files)
│   ├── D/ (4 files)
│   └── F/ (4 files)
├── docs/
│   ├── annotation_guide.md
│   └── calibration_quiz.md
├── scripts/
│   ├── collect_pilot_data.py        ← 새로 만든 것
│   ├── compute_iaa.py                ← 새로 만든 것
│   └── validate_annotations.py       (나중에 추가)
├── config/
│   ├── label_studio_M_template.xml
│   ├── label_studio_D_template.xml
│   └── label_studio_F_template.xml
└── data/
    ├── pilot/ (수집된 데이터)
    └── annotations/ (라벨링 결과)
```

```python
{
  "irf_id": "irf_calc_philosophical_ai",
  "irf_version": "0.0.2",
  "description": "IRF-Calc v0.0.2 – Philosophical Algorithms for AI (LLM fine-tuning + inference controller + DR3 decision layer)",
  "base_model": "your-llm-name-here",
  "target_role": "IRF-philosophical-reasoner",

  "components": ["M", "A", "D", "I", "F", "P"],

  "scoring": {
    "epsilon": 0.001,
    "global_formula": "s_i = (Π_k (epsilon + q_{i,k}))^(1/6)",
    "integer_component": "R_{i,k} = round(10 * q_{i,k})",
    "integer_global": "R_i = round(100 * s_i)",
    "thresholds": {
      "global_min": 0.78,
      "component_min": {
        "M": 0.70,
        "A": 0.70,
        "D": 0.70,
        "I": 0.70,
        "F": 0.70,
        "P": 0.70
      }
    }
  },

  "drift_control": {
    "jsd_max": 0.06,
    "l2_max": 0.04
  },

  "calibration": {
    "corr_ref_min": 0.90,
    "note": "Use arxiv-derived D/I/A + abduction + Socratic benchmarks as reference tasks."
  },

  "training": {
    "strategy": "stagewise_instruction_tuning",
    "stages": [
      {
        "name": "premise_purification",
        "id": "M",
        "task_type": "premise_extraction_and_critique",
        "datasets": [
          "custom:irf_premise_purification_v1"
        ],
        "input_format": {
          "fields": ["question", "context"],
          "output_fields": ["premise_list", "suspect_flags", "M_score"]
        },
        "loss": "cross_entropy + regression(M_score)",
        "metrics": ["M_score_mae", "M_score_corr"],
        "max_steps": 30000
      },
      {
        "name": "hypothesis_generation",
        "id": "A",
        "task_type": "abductive_hypothesis_generation",
        "datasets": [
          "arxiv:causejudger_causelogics",
          "arxiv:gear_abduction",
          "custom:irf_abduction_v1"
        ],
        "input_format": {
          "fields": ["scenario", "observations"],
          "output_fields": ["hypotheses", "plausibility_scores", "A_score"]
        },
        "loss": "sequence_ce + regression(A_score)",
        "metrics": ["A_score_mae", "A_score_corr"],
        "max_steps": 40000
      },
      {
        "name": "deductive_expansion",
        "id": "D",
        "task_type": "logical_consequence_derivation",
        "datasets": [
          "logic:theorem_proving_synthetic",
          "custom:irf_deduction_v1"
        ],
        "input_format": {
          "fields": ["premises", "hypothesis"],
          "output_fields": ["deductive_trace", "D_score"]
        },
        "loss": "sequence_ce + regression(D_score)",
        "metrics": ["D_score_mae", "D_score_corr"],
        "max_steps": 40000
      },
      {
        "name": "inductive_check",
        "id": "I",
        "task_type": "inductive_consistency_check",
        "datasets": [
          "arxiv:inductive_linguistic_reasoning",
          "custom:irf_induction_v1"
        ],
        "input_format": {
          "fields": ["hypothesis", "observations"],
          "output_fields": ["fit_score", "counterexamples", "I_score"]
        },
        "loss": "regression(I_score)",
        "metrics": ["I_score_mae", "I_score_corr"],
        "max_steps": 30000
      },
      {
        "name": "falsification_gate",
        "id": "F",
        "task_type": "falsification_labeling",
        "datasets": [
          "custom:irf_falsification_v1"
        ],
        "input_format": {
          "fields": ["hypothesis", "evidence"],
          "output_fields": ["label", "F_score"]
        },
        "loss": "classification(label) + regression(F_score)",
        "metrics": ["f1_macro", "F_score_corr"],
        "max_steps": 25000
      },
      {
        "name": "paradigm_alignment",
        "id": "P",
        "task_type": "paradigm_consistency_and_shift",
        "datasets": [
          "custom:irf_paradigm_v1",
          "arxiv:socratic_llm_datasets"
        ],
        "input_format": {
          "fields": ["current_paradigm", "new_evidence"],
          "output_fields": ["paradigm_notes", "shift_type", "P_score"]
        },
        "loss": "sequence_ce + regression(P_score)",
        "metrics": ["P_score_mae", "P_score_corr"],
        "max_steps": 25000
      }
    ],
    "joint_finetune": {
      "enabled": true,
      "description": "Final multi-task joint tuning over all IRF components.",
      "sampling_strategy": "proportional_to_dataset_size",
      "max_steps": 60000
    }
  },

  "inference_controller": {
    "loop": {
      "max_cycles": 2,
      "stop_conditions": [
        "exists(h_i: s_i >= thresholds.global_min AND all(q_{i,k} >= thresholds.component_min[k]))",
        "drift_jsd_irf <= drift_control.jsd_max",
        "drift_l2_irf <= drift_control.l2_max"
      ]
    },
    "prompts": {
      "premise_purification": "List all explicit and implicit premises in the question, mark doubtful ones, and output M_score in [0,1].",
      "hypothesis_generation": "Generate 3-5 plausible hypotheses with pros/cons and output A_score in [0,1].",
      "deductive_expansion": "For each hypothesis, derive step-by-step logical consequences and output D_score in [0,1].",
      "inductive_check": "Given data/facts, estimate inductive fit and output I_score in [0,1].",
      "falsification_gate": "Search for killer counterexamples, label each hypothesis as accept/hold/reject, and output F_score in [0,1].",
      "paradigm_alignment": "Describe how the results align/conflict with current paradigm and output P_score in [0,1]."
    }
  },

  "decision_layer": {
    "name": "DR3_decision_protocol",
    "description": "Deductive + Inductive fusion with 3 candidate strategies and conservative final choice.",
    "axes": [
      "realism",
      "stability",
      "conservative_rationality"
    ],
    "weights": {
      "realism": 0.4,
      "stability": 0.4,
      "conservative_rationality": 0.2
    },
    "min_axis_score": 0.6,
    "candidates": 3,
    "mapping": {
      "deductive_sources": ["D", "F"],
      "inductive_sources": ["A", "I"],
      "meta_sources": ["M", "P"]
    },
    "selection_rule": "select argmax_j score_j s.t. all(axis_score_j >= min_axis_score)",
    "fallback_rule": "if no candidate satisfies constraints, abstain_and_request_more_evidence",
    "notes": "D,F는 연역/반증 기반, A,I는 가설/귀납 기반, M,P는 전제/패러다임 기반 메타 정보로 사용한다."
  }
}
```

```json
#!/usr/bin/env python3
"""
IRF-DF-Mini Pilot Data Collection Script

이 스크립트는 GitHub API를 사용하여 code review 코멘트를 크롤링하고,
M/D/F 컴포넌트 학습에 적합한 데이터를 수집합니다.

Requirements:
    pip install requests tqdm

Usage:
    export GITHUB_TOKEN="your_github_personal_access_token"
    python collect_pilot_data.py --component M --count 30
"""

import os
import json
import time
import argparse
from typing import List, Dict, Optional
from datetime import datetime, timedelta

import requests
from tqdm import tqdm

class GitHubDataCollector:
    """GitHub API를 사용한 데이터 수집기"""
    
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("GITHUB_TOKEN")
        if not self.token:
            raise ValueError("GITHUB_TOKEN 환경변수를 설정하거나 token을 전달하세요")
        
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json"
        })
        self.base_url = "https://api.github.com"
    
    def search_repos(self, language: str = "python", min_stars: int = 1000, 
                     max_results: int = 10) -> List[str]:
        """인기 있는 저장소 검색"""
        query = f"language:{language} stars:>{min_stars}"
        url = f"{self.base_url}/search/repositories"
        params = {"q": query, "sort": "stars", "per_page": max_results}
        
        response = self.session.get(url, params=params)
        response.raise_for_status()
        
        repos = response.json()["items"]
        return [repo["full_name"] for repo in repos]
    
    def get_pull_request_reviews(self, repo: str, pr_number: int) -> List[Dict]:
        """특정 PR의 리뷰 코멘트 가져오기"""
        url = f"{self.base_url}/repos/{repo}/pulls/{pr_number}/comments"
        response = self.session.get(url)
        
        if response.status_code != 200:
            return []
        
        return response.json()
    
    def search_prs_with_reviews(self, repo: str, max_prs: int = 50) -> List[Dict]:
        """리뷰가 있는 PR 검색"""
        # 최근 6개월 내 merged PR
        since_date = (datetime.now() - timedelta(days=180)).isoformat()
        url = f"{self.base_url}/repos/{repo}/pulls"
        params = {
            "state": "closed",
            "sort": "updated",
            "direction": "desc",
            "per_page": max_prs
        }
        
        response = self.session.get(url, params=params)
        if response.status_code != 200:
            return []
        
        prs = response.json()
        
        # Merged이고 리뷰가 있는 PR만 필터링
        result = []
        for pr in prs:
            if pr.get("merged_at"):
                result.append({
                    "repo": repo,
                    "pr_number": pr["number"],
                    "title": pr["title"],
                    "merged_at": pr["merged_at"]
                })
        
        return result
    
    def collect_m_data(self, count: int = 30) -> List[Dict]:
        """M (Premise Purification) 데이터 수집
        
        Code review 코멘트에서 '이 변경은 X를 가정한다'는 류의 코멘트 수집
        """
        print(f"🔍 M 컴포넌트 데이터 {count}개 수집 시작...")
        
        # 검색 키워드: assume, premise, given that 등
        keywords = ["assumes", "assuming", "premise", "given that", "this relies on"]
        
        repos = self.search_repos(language="python", min_stars=5000, max_results=5)
        print(f"📦 검색 대상 저장소: {', '.join(repos)}")
        
        collected = []
        
        for repo in tqdm(repos, desc="저장소 탐색"):
            prs = self.search_prs_with_reviews(repo, max_prs=20)
            
            for pr in prs:
                if len(collected) >= count:
                    break
                
                reviews = self.get_pull_request_reviews(repo, pr["pr_number"])
                
                for review in reviews:
                    body = review.get("body", "").lower()
                    
                    # 키워드가 포함된 코멘트만
                    if any(kw in body for kw in keywords):
                        collected.append({
                            "component": "M",
                            "source": "github_pr_review",
                            "repo": repo,
                            "pr_number": pr["pr_number"],
                            "pr_title": pr["title"],
                            "question": f"This code change: {pr['title']}",
                            "context": review["body"],
                            "metadata": {
                                "pr_url": f"https://github.com/{repo}/pull/{pr['pr_number']}",
                                "collected_at": datetime.now().isoformat()
                            }
                        })
                        
                        if len(collected) >= count:
                            break
                
                # Rate limiting 방지
                time.sleep(0.5)
            
            if len(collected) >= count:
                break
        
        print(f"✅ M 데이터 {len(collected)}개 수집 완료")
        return collected
    
    def collect_d_data(self, count: int = 40) -> List[Dict]:
        """D (Deduction) 데이터 수집
        
        Code review에서 논리적 단계를 설명하는 코멘트 수집
        """
        print(f"🔍 D 컴포넌트 데이터 {count}개 수집 시작...")
        
        keywords = ["because", "therefore", "thus", "this means", "which leads to",
                   "step by step", "first", "second", "then"]
        
        repos = self.search_repos(language="python", min_stars=5000, max_results=5)
        collected = []
        
        for repo in tqdm(repos, desc="저장소 탐색"):
            prs = self.search_prs_with_reviews(repo, max_prs=20)
            
            for pr in prs:
                if len(collected) >= count:
                    break
                
                reviews = self.get_pull_request_reviews(repo, pr["pr_number"])
                
                for review in reviews:
                    body = review.get("body", "")
                    
                    # 길이가 적당하고 (100-500자) 키워드 포함
                    if 100 <= len(body) <= 500 and any(kw in body.lower() for kw in keywords):
                        collected.append({
                            "component": "D",
                            "source": "github_pr_review",
                            "repo": repo,
                            "pr_number": pr["pr_number"],
                            "premises": f"Code change in {pr['title']}",
                            "hypothesis": "This change is safe/correct",
                            "response": body,
                            "metadata": {
                                "pr_url": f"https://github.com/{repo}/pull/{pr['pr_number']}",
                                "collected_at": datetime.now().isoformat()
                            }
                        })
                        
                        if len(collected) >= count:
                            break
                
                time.sleep(0.5)
            
            if len(collected) >= count:
                break
        
        print(f"✅ D 데이터 {len(collected)}개 수집 완료")
        return collected
    
    def collect_f_data(self, count: int = 30) -> List[Dict]:
        """F (Falsification) 데이터 수집
        
        'What if', 'edge case', 'what about' 류의 반례 제시 코멘트 수집
        """
        print(f"🔍 F 컴포넌트 데이터 {count}개 수집 시작...")
        
        keywords = ["what if", "edge case", "what about", "but what happens",
                   "counterexample", "fails when", "doesn't work", "breaks"]
        
        repos = self.search_repos(language="python", min_stars=5000, max_results=5)
        collected = []
        
        for repo in tqdm(repos, desc="저장소 탐색"):
            prs = self.search_prs_with_reviews(repo, max_prs=20)
            
            for pr in prs:
                if len(collected) >= count:
                    break
                
                reviews = self.get_pull_request_reviews(repo, pr["pr_number"])
                
                for review in reviews:
                    body = review.get("body", "")
                    
                    if any(kw in body.lower() for kw in keywords):
                        collected.append({
                            "component": "F",
                            "source": "github_pr_review",
                            "repo": repo,
                            "pr_number": pr["pr_number"],
                            "hypothesis": f"This change works correctly: {pr['title']}",
                            "evidence": "Code review discussion",
                            "response": body,
                            "metadata": {
                                "pr_url": f"https://github.com/{repo}/pull/{pr['pr_number']}",
                                "collected_at": datetime.now().isoformat()
                            }
                        })
                        
                        if len(collected) >= count:
                            break
                
                time.sleep(0.5)
            
            if len(collected) >= count:
                break
        
        print(f"✅ F 데이터 {len(collected)}개 수집 완료")
        return collected

def generate_synthetic_data(component: str, count: int) -> List[Dict]:
    """GitHub 데이터가 부족할 경우 synthetic 데이터 생성
    
    나중에 GPT-4로 생성 + human verification 하면 됨
    """
    print(f"⚠️  {component} 컴포넌트: GitHub에서 충분한 데이터를 찾지 못했습니다.")
    print(f"💡 대안: GPT-4로 synthetic 데이터 생성 후 expert verification 권장")
    
    return [
        {
            "component": component,
            "source": "synthetic_placeholder",
            "note": f"이 {count}개 데이터는 GPT-4 생성 + expert verification 필요",
            "generated_at": datetime.now().isoformat()
        }
    ]

def main():
    parser = argparse.ArgumentParser(description="IRF-DF-Mini Pilot 데이터 수집")
    parser.add_argument("--component", choices=["M", "D", "F", "all"], default="all",
                       help="수집할 컴포넌트")
    parser.add_argument("--count", type=int, default=None,
                       help="수집할 데이터 개수 (기본: M=30, D=40, F=30)")
    parser.add_argument("--output", default="pilot_data",
                       help="출력 디렉토리 (기본: pilot_data/)")
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output, exist_ok=True)
    
    try:
        collector = GitHubDataCollector()
        
        if args.component == "all" or args.component == "M":
            m_count = args.count if args.count else 30
            m_data = collector.collect_m_data(m_count)
            
            # 부족하면 synthetic placeholder
            if len(m_data) < m_count:
                m_data.extend(generate_synthetic_data("M", m_count - len(m_data)))
            
            with open(f"{args.output}/M_pilot.json", "w", encoding="utf-8") as f:
                json.dump(m_data, f, indent=2, ensure_ascii=False)
        
        if args.component == "all" or args.component == "D":
            d_count = args.count if args.count else 40
            d_data = collector.collect_d_data(d_count)
            
            if len(d_data) < d_count:
                d_data.extend(generate_synthetic_data("D", d_count - len(d_data)))
            
            with open(f"{args.output}/D_pilot.json", "w", encoding="utf-8") as f:
                json.dump(d_data, f, indent=2, ensure_ascii=False)
        
        if args.component == "all" or args.component == "F":
            f_count = args.count if args.count else 30
            f_data = collector.collect_f_data(f_count)
            
            if len(f_data) < f_count:
                f_data.extend(generate_synthetic_data("F", f_count - len(f_data)))
            
            with open(f"{args.output}/F_pilot.json", "w", encoding="utf-8") as f:
                json.dump(f_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n🎉 데이터 수집 완료! 결과는 {args.output}/ 디렉토리에 저장됨")
        print("\n다음 단계:")
        print("1. 수집된 데이터 품질 확인")
        print("2. Synthetic placeholder를 GPT-4로 생성")
        print("3. Expert가 모든 데이터 검토")
        print("4. Label Studio로 import")
        
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        print("\n문제 해결 팁:")
        print("1. GITHUB_TOKEN이 올바르게 설정되었는지 확인")
        print("2. Token에 'repo' scope 권한이 있는지 확인")
        print("3. Rate limit에 걸렸다면 1시간 후 재시도")

if __name__ == "__main__":
    main()
```

```json
#!/usr/bin/env python3
"""
Inter-Annotator Agreement (IAA) 계산 스크립트

Cohen's Kappa, Fleiss' Kappa, Krippendorff's Alpha를 계산합니다.

Requirements:
    pip install numpy pandas scikit-learn statsmodels

Usage:
    python compute_iaa.py --input annotations.json --output iaa_report.json
"""

import json
import argparse
from typing import List, Dict, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters

class IAACalculator:
    """Inter-Annotator Agreement 계산기"""
    
    def __init__(self, annotations: List[Dict]):
        """
        annotations: [
            {
                "item_id": "M_000001",
                "annotator": "annotator_1",
                "label": "A_better",  # or M_score, etc.
                "component": "M"
            },
            ...
        ]
        """
        self.annotations = annotations
        self.df = pd.DataFrame(annotations)
    
    def cohen_kappa_pairwise(self, annotator1: str, annotator2: str, 
                             component: str = None) -> Dict:
        """두 annotator 간의 Cohen's Kappa 계산"""
        
        df = self.df.copy()
        if component:
            df = df[df["component"] == component]
        
        # 두 annotator의 annotation이 모두 있는 item만 선택
        items_a1 = set(df[df["annotator"] == annotator1]["item_id"])
        items_a2 = set(df[df["annotator"] == annotator2]["item_id"])
        overlap_items = items_a1 & items_a2
        
        if len(overlap_items) == 0:
            return {
                "annotators": [annotator1, annotator2],
                "component": component,
                "overlap_count": 0,
                "kappa": None,
                "error": "No overlapping items"
            }
        
        # 각 annotator의 라벨 추출
        labels_a1 = []
        labels_a2 = []
        
        for item_id in sorted(overlap_items):
            label1 = df[(df["item_id"] == item_id) & 
                       (df["annotator"] == annotator1)]["label"].values[0]
            label2 = df[(df["item_id"] == item_id) & 
                       (df["annotator"] == annotator2)]["label"].values[0]
            labels_a1.append(label1)
            labels_a2.append(label2)
        
        # Cohen's Kappa 계산
        kappa = cohen_kappa_score(labels_a1, labels_a2)
        
        # Agreement rate
        agreement_rate = np.mean([l1 == l2 for l1, l2 in zip(labels_a1, labels_a2)])
        
        return {
            "annotators": [annotator1, annotator2],
            "component": component,
            "overlap_count": len(overlap_items),
            "kappa": float(kappa),
            "agreement_rate": float(agreement_rate),
            "interpretation": self._interpret_kappa(kappa)
        }
    
    def fleiss_kappa_multi(self, component: str = None) -> Dict:
        """여러 annotator 간의 Fleiss' Kappa 계산"""
        
        df = self.df.copy()
        if component:
            df = df[df["component"] == component]
        
        # Item × Annotator matrix 구성
        item_annotator = df.pivot_table(
            index="item_id",
            columns="annotator",
            values="label",
            aggfunc="first"
        )
        
        # 모든 annotator가 라벨링한 item만 사용
        complete_items = item_annotator.dropna()
        
        if len(complete_items) == 0:
            return {
                "component": component,
                "n_items": 0,
                "n_annotators": 0,
                "fleiss_kappa": None,
                "error": "No items with complete annotations"
            }
        
        # Label을 숫자로 변환
        all_labels = df["label"].unique()
        label_to_idx = {label: idx for idx, label in enumerate(sorted(all_labels))}
        
        # Rating matrix 구성: (items × categories)
        n_items = len(complete_items)
        n_categories = len(all_labels)
        rating_matrix = np.zeros((n_items, n_categories))
        
        for i, (item_id, row) in enumerate(complete_items.iterrows()):
            for annotator_label in row.values:
                if pd.notna(annotator_label):
                    rating_matrix[i, label_to_idx[annotator_label]] += 1
        
        # Fleiss' Kappa 계산
        try:
            kappa = fleiss_kappa(rating_matrix, method='fleiss')
        except Exception as e:
            return {
                "component": component,
                "n_items": n_items,
                "n_annotators": len(complete_items.columns),
                "fleiss_kappa": None,
                "error": str(e)
            }
        
        return {
            "component": component,
            "n_items": n_items,
            "n_annotators": len(complete_items.columns),
            "n_categories": n_categories,
            "fleiss_kappa": float(kappa),
            "interpretation": self._interpret_kappa(kappa)
        }
    
    def score_correlation(self, annotator1: str, annotator2: str, 
                         component: str = None) -> Dict:
        """연속 점수 (M_score 등)의 상관관계 계산"""
        
        df = self.df.copy()
        if component:
            df = df[df["component"] == component]
        
        # 'score' 컬럼이 있다고 가정
        if "score" not in df.columns:
            return {
                "error": "No 'score' column found. Use this for M_score, D_score, F_score."
            }
        
        # Overlap items
        items_a1 = set(df[df["annotator"] == annotator1]["item_id"])
        items_a2 = set(df[df["annotator"] == annotator2]["item_id"])
        overlap_items = items_a1 & items_a2
        
        if len(overlap_items) == 0:
            return {"error": "No overlapping items"}
        
        scores_a1 = []
        scores_a2 = []
        
        for item_id in sorted(overlap_items):
            score1 = df[(df["item_id"] == item_id) & 
                       (df["annotator"] == annotator1)]["score"].values[0]
            score2 = df[(df["item_id"] == item_id) & 
                       (df["annotator"] == annotator2)]["score"].values[0]
            scores_a1.append(float(score1))
            scores_a2.append(float(score2))
        
        # Pearson correlation
        correlation = np.corrcoef(scores_a1, scores_a2)[0, 1]
        
        # MAE (Mean Absolute Error)
        mae = np.mean(np.abs(np.array(scores_a1) - np.array(scores_a2)))
        
        return {
            "annotators": [annotator1, annotator2],
            "component": component,
            "overlap_count": len(overlap_items),
            "pearson_correlation": float(correlation),
            "mae": float(mae),
            "interpretation": self._interpret_correlation(correlation)
        }
    
    @staticmethod
    def _interpret_kappa(kappa: float) -> str:
        """Kappa 값 해석 (Landis & Koch, 1977)"""
        if kappa < 0:
            return "Poor (worse than chance)"
        elif kappa < 0.20:
            return "Slight"
        elif kappa < 0.40:
            return "Fair"
        elif kappa < 0.60:
            return "Moderate"
        elif kappa < 0.80:
            return "Substantial"
        else:
            return "Almost Perfect"
    
    @staticmethod
    def _interpret_correlation(r: float) -> str:
        """상관계수 해석"""
        abs_r = abs(r)
        if abs_r < 0.3:
            return "Weak"
        elif abs_r < 0.7:
            return "Moderate"
        else:
            return "Strong"
    
    def generate_report(self, output_file: str = None) -> Dict:
        """전체 IAA 리포트 생성"""
        
        report = {
            "summary": {},
            "pairwise_kappa": [],
            "fleiss_kappa": [],
            "score_correlations": [],
            "recommendations": []
        }
        
        # 고유 annotator, component 추출
        annotators = self.df["annotator"].unique()
        components = self.df["component"].unique()
        
        report["summary"] = {
            "n_annotators": len(annotators),
            "n_components": len(components),
            "n_total_annotations": len(self.df),
            "annotators": list(annotators),
            "components": list(components)
        }
        
        # Pairwise Cohen's Kappa
        print("📊 계산 중: Pairwise Cohen's Kappa...")
        for i, a1 in enumerate(annotators):
            for a2 in annotators[i+1:]:
                for component in components:
                    result = self.cohen_kappa_pairwise(a1, a2, component)
                    if result["kappa"] is not None:
                        report["pairwise_kappa"].append(result)
        
        # Fleiss' Kappa (각 component별)
        print("📊 계산 중: Fleiss' Kappa...")
        for component in components:
            result = self.fleiss_kappa_multi(component)
            if result.get("fleiss_kappa") is not None:
                report["fleiss_kappa"].append(result)
        
        # Score correlation (M_score 등)
        if "score" in self.df.columns:
            print("📊 계산 중: Score Correlations...")
            for i, a1 in enumerate(annotators):
                for a2 in annotators[i+1:]:
                    for component in components:
                        result = self.score_correlation(a1, a2, component)
                        if "error" not in result:
                            report["score_correlations"].append(result)
        
        # Recommendations
        avg_kappa = np.mean([r["kappa"] for r in report["pairwise_kappa"]])
        
        if avg_kappa < 0.60:
            report["recommendations"].append({
                "level": "critical",
                "message": f"평균 Kappa {avg_kappa:.3f} < 0.60. Annotator 재교육 필요.",
                "action": "Guideline 재검토, Golden examples 추가, 1:1 코칭"
            })
        elif avg_kappa < 0.70:
            report["recommendations"].append({
                "level": "warning",
                "message": f"평균 Kappa {avg_kappa:.3f}는 acceptable하지만 개선 가능.",
                "action": "불일치 사례 분석, FAQ 업데이트"
            })
        else:
            report["recommendations"].append({
                "level": "success",
                "message": f"평균 Kappa {avg_kappa:.3f} >= 0.70. 품질 우수!",
                "action": "현재 프로세스 유지"
            })
        
        # 출력
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"✅ 리포트 저장: {output_file}")
        
        return report
    
    def print_summary(self, report: Dict):
        """리포트 요약 출력"""
        print("\n" + "="*60)
        print("📊 Inter-Annotator Agreement Report")
        print("="*60)
        
        summary = report["summary"]
        print(f"\n📌 기본 정보:")
        print(f"  - Annotators: {summary['n_annotators']}명")
        print(f"  - Components: {', '.join(summary['components'])}")
        print(f"  - Total Annotations: {summary['n_total_annotations']}개")
        
        if report["pairwise_kappa"]:
            print(f"\n📊 Pairwise Cohen's Kappa:")
            for result in report["pairwise_kappa"]:
                print(f"  - {result['annotators'][0]} vs {result['annotators'][1]} "
                      f"({result['component']}): κ = {result['kappa']:.3f} "
                      f"({result['interpretation']})")
        
        if report["fleiss_kappa"]:
            print(f"\n📊 Fleiss' Kappa (Multi-rater):")
            for result in report["fleiss_kappa"]:
                print(f"  - {result['component']}: κ = {result['fleiss_kappa']:.3f} "
                      f"({result['interpretation']})")
        
        if report["score_correlations"]:
            print(f"\n📊 Score Correlations:")
            for result in report["score_correlations"]:
                print(f"  - {result['annotators'][0]} vs {result['annotators'][1]} "
                      f"({result['component']}): r = {result['pearson_correlation']:.3f}, "
                      f"MAE = {result['mae']:.3f}")
        
        print(f"\n💡 Recommendations:")
        for rec in report["recommendations"]:
            emoji = {"critical": "🚨", "warning": "⚠️", "success": "✅"}[rec["level"]]
            print(f"  {emoji} {rec['message']}")
            print(f"     → {rec['action']}")
        
        print("\n" + "="*60)

def load_annotations_from_label_studio(json_file: str) -> List[Dict]:
    """Label Studio export JSON을 표준 형식으로 변환"""
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    annotations = []
    
    for task in data:
        item_id = task.get("data", {}).get("item_id", task["id"])
        
        for annotation in task.get("annotations", []):
            annotator = annotation["completed_by"]
            
            # Label 추출 (task에 따라 다를 수 있음)
            result = annotation["result"][0]
            label = result["value"].get("choices", [None])[0]
            
            # Score 추출 (있다면)
            score = result["value"].get("rating")
            
            annotations.append({
                "item_id": str(item_id),
                "annotator": f"annotator_{annotator}",
                "label": label,
                "score": score,
                "component": task.get("data", {}).get("component", "unknown")
            })
    
    return annotations

def main():
    parser = argparse.ArgumentParser(description="IAA 계산")
    parser.add_argument("--input", required=True, help="Annotations JSON 파일")
    parser.add_argument("--output", default="iaa_report.json", help="출력 파일")
    parser.add_argument("--format", choices=["standard", "label_studio"], 
                       default="label_studio",
                       help="입력 파일 형식")
    
    args = parser.parse_args()
    
    # Load annotations
    if args.format == "label_studio":
        annotations = load_annotations_from_label_studio(args.input)
    else:
        with open(args.input, "r", encoding="utf-8") as f:
            annotations = json.load(f)
    
    print(f"📥 {len(annotations)}개 annotation 로드됨")
    
    # Calculate IAA
    calculator = IAACalculator(annotations)
    report = calculator.generate_report(args.output)
    
    # Print summary
    calculator.print_summary(report)

if __name__ == "__main__":
    main()
```

## 1️⃣ `schemas/schema_M_premise_purification.json`

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://example.com/schema_M_premise_purification.json",
  "title": "IRF-DF-Mini M_premise_purification Sample",
  "type": "object",
  "additionalProperties": false,
  "required": ["meta", "task", "input", "label"],
  "properties": {
    "meta": {
      "type": "object",
      "additionalProperties": false,
      "required": ["id", "split"],
      "properties": {
        "id": {
          "type": "string",
          "pattern": "^M_\\d{6}$"
        },
        "version": {
          "type": "string"
        },
        "created_at": {
          "type": "string"
        },
        "source": {
          "type": "object",
          "additionalProperties": false,
          "required": ["type"],
          "properties": {
            "type": {
              "type": "string",
              "enum": [
                "github_pr",
                "textbook",
                "synthetic",
                "proof_log",
                "bug_report",
                "other"
              ]
            },
            "url": {
              "type": "string"
            }
          }
        },
        "annotators": {
          "type": "object",
          "additionalProperties": false,
          "properties": {
            "primary": { "type": "string" },
            "reviewers": {
              "type": "array",
              "items": { "type": "string" }
            }
          }
        },
        "split": {
          "type": "string",
          "enum": ["train", "dev", "test", "golden"]
        }
      }
    },
    "task": {
      "type": "string",
      "const": "M_premise_purification"
    },
    "input": {
      "type": "object",
      "additionalProperties": false,
      "required": ["question", "context", "model_response"],
      "properties": {
        "question": { "type": "string" },
        "context": { "type": "string" },
        "model_response": { "type": "string" }
      }
    },
    "label": {
      "type": "object",
      "additionalProperties": false,
      "required": ["premise_list", "suspect_flags", "M_score", "M_rationale"],
      "properties": {
        "premise_list": {
          "type": "array",
          "items": {
            "type": "object",
            "additionalProperties": false,
            "required": ["text", "source"],
            "properties": {
              "text": { "type": "string" },
              "source": {
                "type": "string",
                "enum": ["explicit", "implicit"]
              }
            }
          }
        },
        "suspect_flags": {
          "type": "array",
          "items": {
            "type": "object",
            "additionalProperties": false,
            "required": ["premise_index", "reason"],
            "properties": {
              "premise_index": {
                "type": "integer",
                "minimum": 0
              },
              "reason": { "type": "string" }
            }
          }
        },
        "missing_premises": {
          "type": "array",
          "items": { "type": "string" }
        },
        "M_score": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 1.0
        },
        "M_rationale": {
          "type": "string"
        }
      }
    }
  }
}

```

---

## 2️⃣ `schemas/schema_D_deduction_pairwise.json`

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://example.com/schema_D_deduction_pairwise.json",
  "title": "IRF-DF-Mini D_deduction_pairwise Sample",
  "type": "object",
  "additionalProperties": false,
  "required": ["meta", "task", "input", "label"],
  "properties": {
    "meta": {
      "type": "object",
      "additionalProperties": false,
      "required": ["id", "split"],
      "properties": {
        "id": {
          "type": "string",
          "pattern": "^D_\\d{6}$"
        },
        "version": {
          "type": "string"
        },
        "created_at": {
          "type": "string"
        },
        "source": {
          "type": "object",
          "additionalProperties": false,
          "required": ["type"],
          "properties": {
            "type": {
              "type": "string",
              "enum": [
                "github_pr",
                "textbook",
                "synthetic",
                "proof_log",
                "bug_report",
                "other"
              ]
            },
            "url": {
              "type": "string"
            }
          }
        },
        "annotators": {
          "type": "object",
          "additionalProperties": false,
          "properties": {
            "primary": { "type": "string" },
            "reviewers": {
              "type": "array",
              "items": { "type": "string" }
            }
          }
        },
        "split": {
          "type": "string",
          "enum": ["train", "dev", "test", "golden"]
        }
      }
    },
    "task": {
      "type": "string",
      "const": "D_deduction_pairwise"
    },
    "input": {
      "type": "object",
      "additionalProperties": false,
      "required": ["premises", "hypothesis", "response_A", "response_B"],
      "properties": {
        "premises": { "type": "string" },
        "hypothesis": { "type": "string" },
        "response_A": { "type": "string" },
        "response_B": { "type": "string" }
      }
    },
    "label": {
      "type": "object",
      "additionalProperties": false,
      "required": ["preference", "D_rationale"],
      "properties": {
        "preference": {
          "type": "string",
          "enum": ["A_better", "B_better", "tie"]
        },
        "D_rationale": {
          "type": "string"
        }
      }
    }
  }
}

```

---

## 3️⃣ `schemas/schema_F_falsification_pairwise.json`

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://example.com/schema_F_falsification_pairwise.json",
  "title": "IRF-DF-Mini F_falsification_pairwise Sample",
  "type": "object",
  "additionalProperties": false,
  "required": ["meta", "task", "input", "label"],
  "properties": {
    "meta": {
      "type": "object",
      "additionalProperties": false,
      "required": ["id", "split"],
      "properties": {
        "id": {
          "type": "string",
          "pattern": "^F_\\d{6}$"
        },
        "version": {
          "type": "string"
        },
        "created_at": {
          "type": "string"
        },
        "source": {
          "type": "object",
          "additionalProperties": false,
          "required": ["type"],
          "properties": {
            "type": {
              "type": "string",
              "enum": [
                "github_pr",
                "textbook",
                "synthetic",
                "proof_log",
                "bug_report",
                "other"
              ]
            },
            "url": {
              "type": "string"
            }
          }
        },
        "annotators": {
          "type": "object",
          "additionalProperties": false,
          "properties": {
            "primary": { "type": "string" },
            "reviewers": {
              "type": "array",
              "items": { "type": "string" }
            }
          }
        },
        "split": {
          "type": "string",
          "enum": ["train", "dev", "test", "golden"]
        }
      }
    },
    "task": {
      "type": "string",
      "const": "F_falsification_pairwise"
    },
    "input": {
      "type": "object",
      "additionalProperties": false,
      "required": ["hypothesis", "evidence", "response_A", "response_B"],
      "properties": {
        "hypothesis": { "type": "string" },
        "evidence": { "type": "string" },
        "response_A": { "type": "string" },
        "response_B": { "type": "string" }
      }
    },
    "label": {
      "type": "object",
      "additionalProperties": false,
      "required": ["preference", "F_rationale"],
      "properties": {
        "preference": {
          "type": "string",
          "enum": ["A_better", "B_better", "tie"]
        },
        "F_rationale": {
          "type": "string"
        },
        "proposed_test_cases": {
          "type": "array",
          "items": { "type": "string" }
        },
        "falsification_verdict": {
          "type": "string",
          "enum": [
            "hypothesis_rejected",
            "weakened",
            "inconclusive"
          ]
        },
        "revised_hypothesis": {
          "type": ["string", "null"]
        }
      }
    }
  }
}

```

---

## 4️⃣ `scripts/validate_annotation.py` (한 파일 완전체)

이 스크립트는:

- `schemas/` 폴더에서 위 3개 스키마를 읽고
- 인자로 받은 경로(디렉토리 or 파일) 아래의 모든 `.json`을 검사해서
- 어떤 파일이 어떤 에러로 깨지는지 출력해준다.

```python
#!/usr/bin/env python
"""
validate_annotation.py

IRF-DF-Mini annotation JSON 파일들을 JSON Schema로 검증하는 스크립트.

사용 예시:
    # 리포 루트에서
    python scripts/validate_annotation.py data/df_mini

    # 특정 파일만
    python scripts/validate_annotation.py data/df_mini/M/M_000001.json
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

try:
    import jsonschema
    from jsonschema import Draft7Validator
except ImportError:
    print(
        "[ERROR] jsonschema 패키지가 필요합니다.\n"
        "pip install jsonschema\n"
    )
    sys.exit(1)

ROOT_DIR = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT_DIR / "schemas"

def load_schema(name: str) -> Dict[str, Any]:
    path = SCHEMA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Schema file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

SCHEMAS = {
    "M_premise_purification": load_schema("schema_M_premise_purification.json"),
    "D_deduction_pairwise": load_schema("schema_D_deduction_pairwise.json"),
    "F_falsification_pairwise": load_schema("schema_F_falsification_pairwise.json"),
}

VALIDATORS = {
    task: Draft7Validator(schema) for task, schema in SCHEMAS.items()
}

def guess_task_from_id(meta: Dict[str, Any]) -> str:
    """
    meta.id prefix로 task를 추측하는 보조 함수.
    - M_****** → M_premise_purification
    - D_****** → D_deduction_pairwise
    - F_****** → F_falsification_pairwise
    """
    id_ = meta.get("id", "")
    if id_.startswith("M_"):
        return "M_premise_purification"
    if id_.startswith("D_"):
        return "D_deduction_pairwise"
    if id_.startswith("F_"):
        return "F_falsification_pairwise"
    return ""

def validate_instance(instance: Dict[str, Any], path: Path) -> bool:
    """
    단일 JSON 객체(instance)에 대해:
      1) task에 맞는 schema 선택
      2) jsonschema로 validate
      3) 추가로 meta.id prefix와 task 일관성 검사

    반환값:
      True  = 유효
      False = 유효하지 않음
    """
    errors = []

    task = instance.get("task")
    meta = instance.get("meta", {})

    if not task:
        errors.append("Missing 'task' field.")
    elif task not in VALIDATORS:
        errors.append(f"Unknown task '{task}'. "
                      f"Expected one of {list(VALIDATORS.keys())}.")

    # meta.id prefix check
    if isinstance(meta, dict) and "id" in meta:
        guessed_task = guess_task_from_id(meta)
        if guessed_task and task and guessed_task != task:
            errors.append(
                f"meta.id '{meta['id']}' implies task '{guessed_task}', "
                f"but 'task' field is '{task}'."
            )

    if errors:
        print(f"[INVALID] {path}")
        for e in errors:
            print(f"  - {e}")
        return False

    # jsonschema validate
    validator = VALIDATORS[task]
    validation_errors = sorted(validator.iter_errors(instance), key=lambda e: e.path)

    if validation_errors:
        print(f"[INVALID] {path}")
        for err in validation_errors:
            # 에러 경로를 사람이 읽기 좋게 표시
            loc = " -> ".join(str(x) for x in err.path) or "(root)"
            print(f"  - At {loc}: {err.message}")
        return False

    # OK
    return True

def iter_json_files(target: Path):
    """
    target이:
      - 디렉토리면: 하위 모든 .json 파일
      - 파일이면: 그 파일 하나
    를 yield
    """
    if target.is_dir():
        yield from target.rglob("*.json")
    else:
        if target.suffix.lower() == ".json":
            yield target

def main(argv=None):
    argv = argv or sys.argv[1:]

    if not argv:
        print(
            "Usage:\n"
            "  python scripts/validate_annotation.py <path-to-json-or-dir>\n\n"
            "예시:\n"
            "  python scripts/validate_annotation.py data/df_mini\n"
            "  python scripts/validate_annotation.py data/df_mini/M/M_000001.json\n"
        )
        return 1

    target = Path(argv[0]).resolve()
    if not target.exists():
        print(f"[ERROR] Target not found: {target}")
        return 1

    total = 0
    ok = 0

    for path in iter_json_files(target):
        total += 1
        try:
            with path.open("r", encoding="utf-8") as f:
                instance = json.load(f)
        except Exception as e:
            print(f"[INVALID] {path}")
            print(f"  - JSON parse error: {e}")
            continue

        if validate_instance(instance, path):
            ok += 1

    print()
    print(f"Validation finished. {ok}/{total} files valid.")

    return 0 if ok == total else 2

if __name__ == "__main__":
    raise SystemExit(main())

```

---

### 📝 사용 요약

1. 위 JSON Schema 3개를 `schemas/`에 저장:
    - `schemas/schema_M_premise_purification.json`
    - `schemas/schema_D_deduction_pairwise.json`
    - `schemas/schema_F_falsification_pairwise.json`
2. Python 스크립트를 `scripts/validate_annotation.py`로 저장
3. 리포 루트에서:

```bash
pip install jsonschema

python scripts/validate_annotation.py data/df_mini
# 또는
python scripts/validate_annotation.py data/df_mini/M/M_000001.json

```

이제부터는 annotator가 만든 JSON이 **스키마/ID/Task 일관성**까지 자동 검증된다.

추가로:

- 다른 필드(예: `falsification_verdict` 강제, `proposed_test_cases` 필수화 등)를 바꾸고 싶으면
    
    → 해당 Schema JSON만 수정하면 파이프라인 전체가 바로 그 규칙을 따른다.
    

## **골든 예제 스키마 + validator** 세트

## `schemas/schema_golden_M_premise_purification.json`

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://example.com/schema_golden_M_premise_purification.json",
  "title": "IRF-DF-Mini Golden Example - M (Premise Purification)",
  "type": "object",
  "additionalProperties": false,
  "required": [
    "golden_id",
    "component",
    "task_type",
    "input",
    "model_response",
    "golden_label"
  ],
  "properties": {
    "golden_id": {
      "type": "string",
      "pattern": "^M_\\d{6}$"
    },
    "component": {
      "type": "string",
      "const": "M"
    },
    "task_type": {
      "type": "string",
      "const": "premise_purification"
    },
    "input": {
      "type": "object",
      "additionalProperties": false,
      "required": ["question", "context"],
      "properties": {
        "question": { "type": "string" },
        "context": { "type": "string" }
      }
    },
    "model_response": {
      "type": "string"
    },
    "golden_label": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "premise_list",
        "suspect_flags",
        "M_score",
        "M_rationale"
      ],
      "properties": {
        "premise_list": {
          "type": "array",
          "minItems": 1,
          "items": {
            "type": "object",
            "additionalProperties": false,
            "required": ["text", "source"],
            "properties": {
              "text": { "type": "string" },
              "source": {
                "type": "string",
                "enum": ["explicit", "implicit"]
              }
            }
          }
        },
        "suspect_flags": {
          "type": "array",
          "items": {
            "type": "object",
            "additionalProperties": false,
            "required": ["premise_index", "reason"],
            "properties": {
              "premise_index": {
                "type": "integer",
                "minimum": 0
              },
              "reason": { "type": "string" }
            }
          }
        },
        "missing_premises": {
          "type": "array",
          "items": { "type": "string" }
        },
        "M_score": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 1.0
        },
        "M_rationale": {
          "type": "string"
        }
      }
    },
    "annotation_notes": {
      "type": "string"
    }
  }
}

```

---

## 2️⃣ `schemas/schema_golden_D_deduction_pairwise.json`

(골든 D 예제 포맷: `golden_id`, `component`, `task_type`, `input`에 premises/hypothesis/response_A/response_B, `golden_label`에 preference + D_rationale.)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://example.com/schema_golden_D_deduction_pairwise.json",
  "title": "IRF-DF-Mini Golden Example - D (Deduction Pairwise)",
  "type": "object",
  "additionalProperties": false,
  "required": [
    "golden_id",
    "component",
    "task_type",
    "input",
    "golden_label"
  ],
  "properties": {
    "golden_id": {
      "type": "string",
      "pattern": "^D_\\d{6}$"
    },
    "component": {
      "type": "string",
      "const": "D"
    },
    "task_type": {
      "type": "string",
      "const": "deduction_pairwise"
    },
    "input": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "premises",
        "hypothesis",
        "response_A",
        "response_B"
      ],
      "properties": {
        "premises": { "type": "string" },
        "hypothesis": { "type": "string" },
        "response_A": { "type": "string" },
        "response_B": { "type": "string" }
      }
    },
    "golden_label": {
      "type": "object",
      "additionalProperties": false,
      "required": ["preference", "D_rationale"],
      "properties": {
        "preference": {
          "type": "string",
          "enum": ["A_better", "B_better", "tie"]
        },
        "D_rationale": {
          "type": "string"
        }
      }
    },
    "annotation_notes": {
      "type": "string"
    }
  }
}

```

---

## 3️⃣ `schemas/schema_golden_F_falsification_pairwise.json`

(F 골든 예제: hypothesis/evidence/response_A/B, golden_label에 preference/F_rationale + optional 테스트케이스/판정/수정가설)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://example.com/schema_golden_F_falsification_pairwise.json",
  "title": "IRF-DF-Mini Golden Example - F (Falsification Pairwise)",
  "type": "object",
  "additionalProperties": false,
  "required": [
    "golden_id",
    "component",
    "task_type",
    "input",
    "golden_label"
  ],
  "properties": {
    "golden_id": {
      "type": "string",
      "pattern": "^F_\\d{6}$"
    },
    "component": {
      "type": "string",
      "const": "F"
    },
    "task_type": {
      "type": "string",
      "const": "falsification_pairwise"
    },
    "input": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "hypothesis",
        "evidence",
        "response_A",
        "response_B"
      ],
      "properties": {
        "hypothesis": { "type": "string" },
        "evidence": { "type": "string" },
        "response_A": { "type": "string" },
        "response_B": { "type": "string" }
      }
    },
    "golden_label": {
      "type": "object",
      "additionalProperties": false,
      "required": ["preference", "F_rationale"],
      "properties": {
        "preference": {
          "type": "string",
          "enum": ["A_better", "B_better", "tie"]
        },
        "F_rationale": {
          "type": "string"
        },
        "proposed_test_cases": {
          "type": "array",
          "items": { "type": "string" }
        },
        "falsification_verdict": {
          "type": "string",
          "enum": [
            "hypothesis_rejected",
            "weakened",
            "inconclusive"
          ]
        },
        "revised_hypothesis": {
          "type": ["string", "null"]
        }
      }
    },
    "annotation_notes": {
      "type": "string"
    }
  }
}

```

---

## 4️⃣ `scripts/validate_annotation.py` (골든 예제 검증용 완전체)

이 스크립트는:

- `schemas/` 폴더에서 위 3개 스키마를 읽고
- 인자로 받은 경로(파일 or 디렉토리) 아래 `.json` 전부 읽어서
- `component` + `task_type` 조합에 맞는 스키마로 validate
    
    (형식/필드 누락/타입 에러/ID 패턴 등 다 잡아줌)
    

```python
#!/usr/bin/env python3
"""
IRF-DF-Mini Golden Example Validator

- golden_examples/M/*.json
- golden_examples/D/*.json
- golden_examples/F/*.json

이 파일들이 schema_golden_*.json에 맞게 잘 작성되었는지 검증한다.

Usage:
    # 리포 루트에서
    python scripts/validate_annotation.py golden_examples

    # 특정 파일만
    python scripts/validate_annotation.py golden_examples/M/M_000001_browser_premise.json
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import jsonschema
    from jsonschema import Draft7Validator
except ImportError:
    print(
        "[ERROR] jsonschema 패키지가 필요합니다.\n"
        "설치:\n"
        "    pip install jsonschema\n"
    )
    sys.exit(1)

ROOT_DIR = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT_DIR / "schemas"

def load_schema(name: str) -> Dict[str, Any]:
    """schemas/ 아래 JSON Schema 로드"""
    path = SCHEMA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Schema file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

# 스키마 로드
SCHEMAS = {
    ("M", "premise_purification"): load_schema("schema_golden_M_premise_purification.json"),
    ("D", "deduction_pairwise"): load_schema("schema_golden_D_deduction_pairwise.json"),
    ("F", "falsification_pairwise"): load_schema("schema_golden_F_falsification_pairwise.json"),
}

VALIDATORS = {
    key: Draft7Validator(schema) for key, schema in SCHEMAS.items()
}

def pick_validator(component: str, task_type: str) -> Optional[Draft7Validator]:
    """component + task_type 조합으로 validator 선택"""
    return VALIDATORS.get((component, task_type))

def validate_instance(instance: Dict[str, Any], path: Path) -> bool:
    """
    단일 JSON 골든 예제를 검증:
      1) component / task_type 존재 확인
      2) 올바른 스키마 선택
      3) jsonschema로 validate

    반환:
      True  = 유효
      False = 유효하지 않음
    """
    errors = []

    component = instance.get("component")
    task_type = instance.get("task_type")
    golden_id = instance.get("golden_id")

    if not golden_id:
        errors.append("Missing 'golden_id' field.")

    if not component:
        errors.append("Missing 'component' field.")
    if not task_type:
        errors.append("Missing 'task_type' field.")

    validator = None
    if component and task_type:
        validator = pick_validator(component, task_type)
        if validator is None:
            valid_keys = ", ".join(f"{c}/{t}" for (c, t) in VALIDATORS.keys())
            errors.append(
                f"Unsupported component/task_type combination: '{component}/{task_type}'. "
                f"지원되는 조합: {valid_keys}"
            )

    if errors:
        print(f"[INVALID] {path}")
        for e in errors:
            print(f"  - {e}")
        return False

    # jsonschema로 실제 필드/타입 검증
    validation_errors = sorted(validator.iter_errors(instance), key=lambda e: e.path)

    if validation_errors:
        print(f"[INVALID] {path}")
        for err in validation_errors:
            loc = " -> ".join(str(x) for x in err.path) or "(root)"
            print(f"  - At {loc}: {err.message}")
        return False

    # OK
    return True

def iter_json_files(target: Path):
    """
    target이:
      - 디렉토리면: 하위 모든 .json 파일
      - 파일이면: 그 파일 하나
    를 yield
    """
    if target.is_dir():
        yield from target.rglob("*.json")
    else:
        if target.suffix.lower() == ".json":
            yield target

def main(argv=None):
    argv = argv or sys.argv[1:]

    if not argv:
        print(
            "Usage:\n"
            "  python scripts/validate_annotation.py <path-to-json-or-dir>\n\n"
            "예시:\n"
            "  python scripts/validate_annotation.py golden_examples\n"
            "  python scripts/validate_annotation.py golden_examples/M/M_000001_browser_premise.json\n"
        )
        return 1

    target = Path(argv[0]).resolve()
    if not target.exists():
        print(f"[ERROR] Target not found: {target}")
        return 1

    total = 0
    ok = 0

    for path in iter_json_files(target):
        total += 1
        try:
            with path.open("r", encoding="utf-8") as f:
                instance = json.load(f)
        except Exception as e:
            print(f"[INVALID] {path}")
            print(f"  - JSON parse error: {e}")
            continue

        if validate_instance(instance, path):
            ok += 1
            print(f"[OK]      {path}")

    print()
    print(f"Validation finished. {ok}/{total} files valid.")

    return 0 if ok == total else 2

if __name__ == "__main__":
    raise SystemExit(main())

```

---

### 🔌 바로 쓰는 방법 정리

프로젝트 루트 구조 예:

```
irf-df-mini-toolkit/
├── schemas/
│   ├── schema_golden_M_premise_purification.json
│   ├── schema_golden_D_deduction_pairwise.json
│   └── schema_golden_F_falsification_pairwise.json
├── golden_examples/
│   ├── M/
│   │   └── M_000001_browser_premise.json
│   ├── D/
│   └── F/
└── scripts/
    └── validate_annotation.py

```

설치 + 실행:

```bash
pip install jsonschema

# 리포 루트에서
python scripts/validate_annotation.py golden_examples
# 또는
python scripts/validate_annotation.py golden_examples/M/M_000001_browser_premise.json

```

이제 **골든 12개 + 이후 추가 골든들**을 전부 이 스키마로 자동 검사할 수 있음.

## **골든 → 학습 포맷**, 그리고 **Label Studio → 2차 검증**까지 “완전한 파이프라인”

## 1️⃣ `convert_golden_to_train.py`

역할:

- `golden_examples/M|D|F/*.json` (이미 우리가 정한 golden schema 준수) →
    
    `data/train/train_M.jsonl`, `train_D.jsonl`, `train_F.jsonl` 로 변환
    
- **M**: supervised 학습용 (`instruction` + `target` + 구조화 label)
- **D/F**: pairwise preference 학습용 (랭킹/선호 모델, 또는 LLM instruction-finetune)

```python
#!/usr/bin/env python3
"""
convert_golden_to_train.py

골든 예제(M/D/F)를 IRF-DF-Mini 내부 학습 포맷(JSONL)으로 변환한다.

입력 구조(리포 루트 기준):
  golden_examples/
    M/*.json  (M_000001_*.json ...)
    D/*.json
    F/*.json
  schemas/
    schema_golden_M_premise_purification.json
    schema_golden_D_deduction_pairwise.json
    schema_golden_F_falsification_pairwise.json

출력:
  data/train/train_M.jsonl
  data/train/train_D.jsonl
  data/train/train_F.jsonl

요구 라이브러리:
  pip install jsonschema
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple

import jsonschema
from jsonschema import Draft7Validator

ROOT_DIR = Path(__file__).resolve().parents[1]
GOLDEN_DIR = ROOT_DIR / "golden_examples"
SCHEMA_DIR = ROOT_DIR / "schemas"
TRAIN_DIR = ROOT_DIR / "data" / "train"

def load_schema(name: str) -> Dict[str, Any]:
    path = SCHEMA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Schema file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

# 스키마 로드 (이전에 정의한 것과 동일한 파일명 사용)
SCHEMA_M = load_schema("schema_golden_M_premise_purification.json")
SCHEMA_D = load_schema("schema_golden_D_deduction_pairwise.json")
SCHEMA_F = load_schema("schema_golden_F_falsification_pairwise.json")

VALIDATORS = {
    ("M", "premise_purification"): Draft7Validator(SCHEMA_M),
    ("D", "deduction_pairwise"): Draft7Validator(SCHEMA_D),
    ("F", "falsification_pairwise"): Draft7Validator(SCHEMA_F),
}

def validate_golden(instance: Dict[str, Any]) -> Tuple[bool, str]:
    """골든 예제가 스키마를 만족하는지 검증."""
    comp = instance.get("component")
    task_type = instance.get("task_type")
    key = (comp, task_type)

    if key not in VALIDATORS:
        return False, f"Unsupported component/task_type: {comp}/{task_type}"

    validator = VALIDATORS[key]
    errors = sorted(validator.iter_errors(instance), key=lambda e: e.path)
    if errors:
        msg_lines = []
        for err in errors:
            loc = " -> ".join(str(x) for x in err.path) or "(root)"
            msg_lines.append(f"At {loc}: {err.message}")
        return False, "; ".join(msg_lines)

    return True, ""

def convert_M(instance: Dict[str, Any]) -> Dict[str, Any]:
    """
    M 골든 예제 → 학습 포맷

    내부 포맷 예:
    {
      "id": "M_000001",
      "component": "M",
      "task_type": "premise_purification",
      "input": {
        "question": "...",
        "context": "...",
        "model_response": "..."
      },
      "labels": {... golden_label ...},
      "training_example": {
        "instruction": "...",
        "target": "..."  # JSON string 또는 자연어 요약
      }
    }
    """
    golden_id = instance["golden_id"]
    inp = instance["input"]
    label = instance["golden_label"]

    instruction = (
        "You are the M (Premise Purification) component of IRF-DF-Mini.\n"
        "Given the question, context, and model_response, list:\n"
        "1) explicit premises\n"
        "2) implicit premises\n"
        "3) which premises are suspect and why\n\n"
        "Return a JSON object with keys: premise_list, suspect_flags, M_score.\n\n"
        f"Question: {inp['question']}\n\n"
        f"Context: {inp['context']}\n\n"
        f"Model response:\n{instance['model_response']}\n"
    )

    # target은 golden_label을 JSON 문자열로 넣어준다
    target = json.dumps(
        {
            "premise_list": label["premise_list"],
            "suspect_flags": label["suspect_flags"],
            "M_score": label["M_score"],
        },
        ensure_ascii=False,
        indent=2,
    )

    return {
        "id": golden_id,
        "component": "M",
        "task_type": "premise_purification",
        "input": {
            "question": inp["question"],
            "context": inp["context"],
            "model_response": instance["model_response"],
        },
        "labels": label,
        "training_example": {
            "instruction": instruction,
            "target": target,
        },
    }

def convert_D(instance: Dict[str, Any]) -> Dict[str, Any]:
    """
    D 골든 예제 → pairwise 학습 포맷

    내부 포맷 예:
    {
      "id": "D_000001",
      "component": "D",
      "task_type": "deduction_pairwise",
      "prompt": {...},
      "response_A": "...",
      "response_B": "...",
      "preference": "A_better",
      "rationale": "..."
    }
    """
    golden_id = instance["golden_id"]
    inp = instance["input"]
    label = instance["golden_label"]

    return {
        "id": golden_id,
        "component": "D",
        "task_type": "deduction_pairwise",
        "prompt": {
            "premises": inp["premises"],
            "hypothesis": inp["hypothesis"],
        },
        "response_A": inp["response_A"],
        "response_B": inp["response_B"],
        "preference": label["preference"],
        "rationale": label["D_rationale"],
    }

def convert_F(instance: Dict[str, Any]) -> Dict[str, Any]:
    """
    F 골든 예제 → pairwise 학습 포맷

    내부 포맷 예:
    {
      "id": "F_000001",
      "component": "F",
      "task_type": "falsification_pairwise",
      "prompt": {...},
      "response_A": "...",
      "response_B": "...",
      "preference": "B_better",
      "rationale": "...",
      "extra": {...}
    }
    """
    golden_id = instance["golden_id"]
    inp = instance["input"]
    label = instance["golden_label"]

    # optional fields 안전하게 빼기
    extra = {
        "proposed_test_cases": label.get("proposed_test_cases", []),
        "falsification_verdict": label.get("falsification_verdict"),
        "revised_hypothesis": label.get("revised_hypothesis"),
    }

    return {
        "id": golden_id,
        "component": "F",
        "task_type": "falsification_pairwise",
        "prompt": {
            "hypothesis": inp["hypothesis"],
            "evidence": inp["evidence"],
        },
        "response_A": inp["response_A"],
        "response_B": inp["response_B"],
        "preference": label["preference"],
        "rationale": label["F_rationale"],
        "extra": extra,
    }

def jsonl_write(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")

def main():
    print("🔁 Converting golden examples → train format ...")

    m_records = []
    d_records = []
    f_records = []

    # M
    for path in sorted((GOLDEN_DIR / "M").glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            instance = json.load(f)
        ok, msg = validate_golden(instance)
        if not ok:
            raise ValueError(f"[M] Invalid golden {path}: {msg}")
        m_records.append(convert_M(instance))

    # D
    for path in sorted((GOLDEN_DIR / "D").glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            instance = json.load(f)
        ok, msg = validate_golden(instance)
        if not ok:
            raise ValueError(f"[D] Invalid golden {path}: {msg}")
        d_records.append(convert_D(instance))

    # F
    for path in sorted((GOLDEN_DIR / "F").glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            instance = json.load(f)
        ok, msg = validate_golden(instance)
        if not ok:
            raise ValueError(f"[F] Invalid golden {path}: {msg}")
        f_records.append(convert_F(instance))

    # JSONL로 저장
    jsonl_write(TRAIN_DIR / "train_M.jsonl", m_records)
    jsonl_write(TRAIN_DIR / "train_D.jsonl", d_records)
    jsonl_write(TRAIN_DIR / "train_F.jsonl", f_records)

    print(f"✅ M: {len(m_records)} → {TRAIN_DIR/'train_M.jsonl'}")
    print(f"✅ D: {len(d_records)} → {TRAIN_DIR/'train_D.jsonl'}")
    print(f"✅ F: {len(f_records)} → {TRAIN_DIR/'train_F.jsonl'}")
    print("🎉 Done. 이제 이 JSONL을 바로 파인튜닝 파이프라인에서 사용할 수 있음.")

if __name__ == "__main__":
    main()

```

---

## 2️⃣ Label Studio export → 2차 검증 validator

이 스크립트는:

1. Label Studio export JSON 로드
2. 우리가 정의한 템플릿(텍스트 영역 이름) 기준으로:
    - M: `explicit_premises`, `implicit_premises`, `suspect_flags`, `m_score`, `m_rationale`
    - D/F: `preference`, `rationale`
3. 각 annotation을 **“golden과 동일한 구조”**로 변환
4. 이미 만들어둔 `schema_golden_*.json`으로 validate
    
    → 구조/스코어 범위/필드 누락 모두 잡는 2차 필터
    

이렇게 하면:

- Annotator들이 Label Studio에서 이상하게 입력한 케이스
- 점수 범위 잘못, suspect_flags 형식 이상 등

을 전부 자동으로 걸러낼 수 있다.

```python
#!/usr/bin/env python3
"""
validate_labelstudio_annotations.py

Label Studio export JSON을 읽어서:
  1) M/D/F 작업을 golden-like internal 구조로 변환
  2) 기존 schema_golden_*.json으로 validate (2차 검증)

Usage:
    python scripts/validate_labelstudio_annotations.py \
        --input data/annotations/ls_export.json

주의:
  - Label Studio 템플릿은 우리가 제안한 XML 기준
    (explicit_premises / implicit_premises / suspect_flags / m_score / m_rationale / preference / rationale)
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

import jsonschema
from jsonschema import Draft7Validator

ROOT_DIR = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT_DIR / "schemas"

def load_schema(name: str) -> Dict[str, Any]:
    path = SCHEMA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Schema file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

SCHEMA_M = load_schema("schema_golden_M_premise_purification.json")
SCHEMA_D = load_schema("schema_golden_D_deduction_pairwise.json")
SCHEMA_F = load_schema("schema_golden_F_falsification_pairwise.json")

VALIDATORS = {
    ("M", "premise_purification"): Draft7Validator(SCHEMA_M),
    ("D", "deduction_pairwise"): Draft7Validator(SCHEMA_D),
    ("F", "falsification_pairwise"): Draft7Validator(SCHEMA_F),
}

def _split_lines(value: Any) -> List[str]:
    """Label Studio textarea → line list 로 normalize."""
    if isinstance(value, list) and value:
        # 보통 ["line1\nline2"] 형태
        text = value[0]
    elif isinstance(value, str):
        text = value
    else:
        return []
    return [line.strip() for line in text.splitlines() if line.strip()]

def _parse_suspect_flags(lines: List[str]) -> List[Dict[str, Any]]:
    """
    suspect_flags textarea:
      "0: 경험적으로 의심 - ..." 형식 → [{"premise_index": 0, "reason": "..."}]
    """
    results = []
    for line in lines:
        if ":" not in line:
            continue
        idx_str, reason = line.split(":", 1)
        idx_str = idx_str.strip()
        reason = reason.strip()
        try:
            idx = int(idx_str)
        except ValueError:
            # 인덱스 파싱 실패 → 스킵 or log
            continue
        if reason:
            results.append({"premise_index": idx, "reason": reason})
    return results

def ls_task_to_golden_M(task: Dict[str, Any], ann: Dict[str, Any]) -> Dict[str, Any]:
    """
    Label Studio M task + annotation → golden-like M object
    (schema_golden_M_premise_purification.json 준수)
    """
    data = task.get("data", {})
    results = ann.get("result", [])

    # result들은 name 필드를 기준으로 찾아서 매핑
    value_map = {}
    for r in results:
        name = r.get("from_name") or r.get("name")
        if not name:
            continue
        value_map[name] = r.get("value", {})

    explicit_lines = _split_lines(value_map.get("explicit_premises", {}).get("text"))
    implicit_lines = _split_lines(value_map.get("implicit_premises", {}).get("text"))
    suspect_lines = _split_lines(value_map.get("suspect_flags", {}).get("text"))

    # premise_list 구성
    premise_list = []
    for t in explicit_lines:
        premise_list.append({"text": t, "source": "explicit"})
    for t in implicit_lines:
        premise_list.append({"text": t, "source": "implicit"})

    suspect_flags = _parse_suspect_flags(suspect_lines)

    # m_score: Rating → 0~10 star를 0~1로 normalize (혹은 그냥 0~10 그대로 써도 되지만
    # schema는 0~1을 기대하므로 간단히 10으로 나눈다)
    m_score_raw = value_map.get("m_score", {}).get("rating")
    if m_score_raw is None:
        m_score = 0.0
    else:
        m_score = float(m_score_raw) / 10.0

    m_rationale = ""
    if "m_rationale" in value_map:
        texts = value_map["m_rationale"].get("text")
        lines = _split_lines(texts)
        m_rationale = "\n".join(lines)

    golden_like = {
        "golden_id": str(task.get("id", data.get("item_id", "M_unknown"))),
        "component": "M",
        "task_type": "premise_purification",
        "input": {
            "question": data.get("question", ""),
            "context": data.get("context", ""),
        },
        "model_response": data.get("model_response", ""),
        "golden_label": {
            "premise_list": premise_list,
            "suspect_flags": suspect_flags,
            "M_score": m_score,
            "M_rationale": m_rationale,
        },
        "annotation_notes": f"from_label_studio_annotator_{ann.get('completed_by')}",
    }

    return golden_like

def ls_task_to_golden_D_or_F(
    task: Dict[str, Any],
    ann: Dict[str, Any],
    component: str,
) -> Dict[str, Any]:
    """
    Label Studio D/F task → golden-like D/F object.

    data 필드 예:
      component: "D" or "F"
      task_type: "deduction_pairwise" / "falsification_pairwise"
      premises/hypothesis/evidence/response_A/response_B ...

    annotation:
      preference (Choices), rationale (TextArea)
    """
    data = task.get("data", {})
    results = ann.get("result", [])

    value_map = {}
    for r in results:
        name = r.get("from_name") or r.get("name")
        if not name:
            continue
        value_map[name] = r.get("value", {})

    # preference: Choices → value list 중 첫 번째
    pref_vals = value_map.get("preference", {}).get("choices", [])
    preference = pref_vals[0] if pref_vals else None

    # rationale
    rationale_lines = _split_lines(value_map.get("rationale", {}).get("text"))
    rationale = "\n".join(rationale_lines)

    task_type = data.get("task_type")
    if component == "D":
        golden_like = {
            "golden_id": str(task.get("id", data.get("item_id", "D_unknown"))),
            "component": "D",
            "task_type": task_type or "deduction_pairwise",
            "input": {
                "premises": data.get("premises", ""),
                "hypothesis": data.get("hypothesis", ""),
                "response_A": data.get("response_A", ""),
                "response_B": data.get("response_B", ""),
            },
            "golden_label": {
                "preference": preference,
                "D_rationale": rationale,
            },
            "annotation_notes": f"from_label_studio_annotator_{ann.get('completed_by')}",
        }
    else:  # F
        golden_like = {
            "golden_id": str(task.get("id", data.get("item_id", "F_unknown"))),
            "component": "F",
            "task_type": task_type or "falsification_pairwise",
            "input": {
                "hypothesis": data.get("hypothesis", ""),
                "evidence": data.get("evidence", ""),
                "response_A": data.get("response_A", ""),
                "response_B": data.get("response_B", ""),
            },
            "golden_label": {
                "preference": preference,
                "F_rationale": rationale,
            },
            "annotation_notes": f"from_label_studio_annotator_{ann.get('completed_by')}",
        }

    return golden_like

def validate_with_schema(instance: Dict[str, Any]) -> Tuple[bool, str]:
    comp = instance.get("component")
    task_type = instance.get("task_type")
    key = (comp, task_type)

    if key not in VALIDATORS:
        return False, f"Unsupported component/task_type: {comp}/{task_type}"

    validator = VALIDATORS[key]
    errors = sorted(validator.iter_errors(instance), key=lambda e: e.path)
    if errors:
        msgs = []
        for err in errors:
            loc = " -> ".join(str(x) for x in err.path) or "(root)"
            msgs.append(f"At {loc}: {err.message}")
        return False, "; ".join(msgs)
    return True, ""

def main():
    parser = argparse.ArgumentParser(
        description="Validate Label Studio export using golden schemas"
    )
    parser.add_argument(
        "--input", required=True, help="Label Studio export JSON file"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="(optional) 변환된 golden-like annotation을 저장할 JSONL 경로",
    )
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    converted: List[Dict[str, Any]] = []
    total = 0
    invalid = 0

    for task in data:
        comp = task.get("data", {}).get("component")
        if not comp:
            # component 없으면 스킵 (또는 로그)
            continue

        for ann in task.get("annotations", []):
            total += 1
            if comp == "M":
                inst = ls_task_to_golden_M(task, ann)
            elif comp in ("D", "F"):
                inst = ls_task_to_golden_D_or_F(task, ann, component=comp)
            else:
                print(f"[WARN] Unknown component {comp}, task id {task.get('id')}")
                continue

            ok, msg = validate_with_schema(inst)
            if not ok:
                invalid += 1
                print(f"[INVALID] task_id={task.get('id')} annotator={ann.get('completed_by')}")
                print(f"  - {msg}")
            else:
                converted.append(inst)

    print()
    print(f"총 annotation: {total}")
    print(f"유효(valid): {total - invalid}")
    print(f"무효(invalid): {invalid}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for inst in converted:
                f.write(json.dumps(inst, ensure_ascii=False))
                f.write("\n")
        print(f"✅ 변환된 golden-like annotation {len(converted)}개를 {out_path} 에 저장")

    # 실패 비율이 높으면 non-zero exit code
    if invalid > 0:
        print("⚠️ 일부 annotation이 schema를 통과하지 못했습니다. 위 로그를 확인하세요.")
        exit(2)
    else:
        print("🎉 모든 annotation이 schema 검증을 통과했습니다.")
        exit(0)

if __name__ == "__main__":
    main()

```

---

## 🔗 정리: 파이프라인 흐름

1. **골든 제작**
    - `golden_examples/M|D|F/*.json` (이미 있음)
    - `scripts/validate_annotation.py golden_examples` → 구조 OK 확인
2. **골든 → 학습 포맷**
    - `python scripts/convert_golden_to_train.py`
    - → `data/train/train_M.jsonl`, `train_D.jsonl`, `train_F.jsonl`
3. **Label Studio 라벨링**
    - export JSON: `data/annotations/ls_export_2025-11-14.json`
4. **Label Studio 결과 2차 검증 + golden-like 변환**
    - `python scripts/validate_labelstudio_annotations.py \ --input data/annotations/ls_export_2025-11-14.json \ --output data/annotations/ls_valid_2025-11-14.jsonl`
5. (옵션) `ls_valid_*.jsonl` → train 포맷으로 합치는 작은 스크립트 추가해서
    
    골든 + 실 annotation 섞어서 full DF-Mini 파인튜닝.