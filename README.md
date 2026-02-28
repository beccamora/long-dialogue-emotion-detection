## Long Dialogue Emotion Detection — Reproduction Study

This project presents a simplified reproduction and critical analysis of *Long Dialogue Emotion Detection Based on Commonsense Knowledge Graph Guidance* by Nie et al. (2023).

The original work proposes a growing graph framework that integrates dialogue structure, commonsense knowledge, latent topic modeling, and transformer-based classification to improve emotion detection in long multi-turn conversations.

### Objective

Rather than fully replicating the computationally expensive architecture, this project focuses on:

- Implementing a lightweight and reproducible version under realistic GPU constraints  
- Evaluating the contribution of each architectural component  
- Analyzing the trade-off between model complexity and empirical performance
  
---

### Model Variants

To isolate architectural impact, three variants were implemented:

- **T1 — Text-only baseline** (RoBERTa + linear classifier)
- **T2 — Text + Commonsense** (static ATOMIC retrieval and fusion)
- **T3 — Full simplified model**
  - Static dialogue graph (GCN)
  - Commonsense integration
  - Attention-based latent topic aggregation

Additional ablation studies removed graph and topic modules to measure their independent contributions.

---

### Key Findings

- The text-only baseline outperformed the full simplified model under constrained training.
- Static commonsense integration provided limited gains.
- Static graph modeling struggled with long-range emotional dependencies.
- Increased architectural complexity introduced optimization instability.
- In some cases, removing graph and topic modules improved performance.

**Main insight:**  
Architectural sophistication does not guarantee better results. Careful ablation and empirical validation are essential when evaluating complex NLP systems.

---

### Contributions

- Resource-aware reimplementation of a knowledge-guided dialogue model  
- Component-level ablation analysis  
- Confusion-matrix–based error analysis  
- Evaluation of complexity vs. performance trade-offs  
- Practical assessment of graph- and knowledge-based emotion modeling  

---

### Takeaway

This project emphasizes the importance of balancing expressiveness, efficiency, and optimization stability in long-context NLP systems. It reflects a research-driven approach to model evaluation rather than performance chasing.

### Reference 
Nie, W., Bao, Y., Zhao, Y., & Liu, A. Long Dialogue Emotion Detection Based on Commonsense Knowledge Graph Guidance. IEEE Transactions on Multimedia, 2024.
