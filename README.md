# Training-Metrics

This repository contains definition of and explorations with new neural network training metrics that frame learning as 
a guided traversal through parameter space. [`Training Metrics.pdf`](./Training%20Metrics.pdf) defines the framework in
detail, as it has been implemented in my main 
[`Trainable Template Tensor Network (TTTN)`](https://github.com/benmeyersUSC/TTTN) C++ machine learning library. This 
repository has a section of C++-based explorations using `TTTN`, dynamically pulling my library from GitHub. I also 
do some quick, vanilla PyTorch exploration of the explanatory power of this framework in 
[`Jupyter Exploration`](./Jupyter%20Exploration). 

## [`Jupyter Exploration`](./Jupyter%20Exploration)

### [`MNIST with Metrics`](Jupyter%20Exploration/MNIST_Metrics_Sweep.ipynb)
#### Background
This file, first and foremost, highlights the incredible leverage that a LLMs like **Claude** give a researcher. After
weeks of development in my [`TTTN`](https://github.com/benmeyersUSC/TTTN) 
library, producing the code and explanatory document ([`Training Metrics.pdf`](./Training%20Metrics.pdf)) for this new
framework, 30 minutes of iteration and discussion with **Claude** have given me a real interesting artifact. The work
to be done was very well described --- in my prompts and in the documents I provided --- and **Claude** realized exactly
what was desired in this preliminary test of plausibility. To just get this cheap signal of the usefulness of my framework,
it would have taken me 2-10x more time; now I can learn from this view into the metrics (for instance, the qualification
of efficiency by actual training loss shed was only visible as obvious when I began observing efficiency across Batch sizes). 
I now not only have something to show for the development of this framework, but I have upstream changes to make to 
[`TTTN`](https://github.com/benmeyersUSC/TTTN) and an enriched intuition about what
is being measured! 

I strongly encourage anybody with describable curiosity to aggressively engage with LLMs. For the work you care about owning
(like [`TTTN`](https://github.com/benmeyersUSC/TTTN) or its child C++ projects, for me), write it all yourself. Struggling
through those steps are what you want to own; but to surface an idea, to test a framework, to try to break your system, 
using an LLM is an insightful and deeply enjoying experience. 

#### Insights

The sweep trains a small feedforward network (784 → 128 → ReLU → 10) on MNIST across eight batch sizes — from 0.1% to 100% of the 60k training set — measuring η1, η2, loss, accuracy, and **qualified efficiency** (η1 × fractional test-loss improvement) at each configuration.

---

##### The Surface-Level View: Everything Trades Off

![Trajectory metrics vs batch fraction](Jupyter%20Exploration/figs/5a_final_metrics_vs_batch_fraction.png)

The top row tells the standard story. η1 rises monotonically with batch size — large batches produce low-variance gradients, so parameters reverse direction less often and gross path length shrinks toward net displacement. Test loss and accuracy tell the opposite story: small batches generalize dramatically better. The Pareto scatter (center-bottom) makes the conflict geometric: **no single batch size is simultaneously low-loss and high-η1**.

But the bottom-right panel — qualified efficiency — is the first sign that the framework offers something new. It peaks sharply at an intermediate batch size (~10%), then collapses in both directions. The metric is doing what loss and η1 alone cannot: it rewards efficient motion *that actually went somewhere*.

---

##### η1 vs η2: Every Parameter Marches Alone

![η1 vs η2 gap](Jupyter%20Exploration/figs/5b_eta1_eta2_gap.png)

η2 (the geodesic / L2 efficiency) is essentially flat near zero across the entire sweep while η1 varies from 0.10 to 0.93. The shaded gap between them is enormous and grows with batch size. What this reveals: even when individual parameters are moving very purposefully (η1 high), they are moving in *incoherent directions relative to each other* — the network-level displacement vector is tiny compared to the sum of individual movements. There is no regime where the network moves as a coordinated unit. Small batches are chaotic in both magnitude and direction; large batches are locally orderly but globally incoherent. Coordinated, purposeful motion — the thing η2 would reward — appears to be the hardest regime to achieve.

---

##### Inside the Network: The Bimodal Efficiency Split

![Per-parameter η_i distributions](Jupyter%20Exploration/figs/5c_per_param_eta_distributions.png)

Network-scalar η1 is a mean. These histograms show what that mean hides. At bs=60, the distribution is broad and flat — the population of per-parameter efficiencies spans the full [0, 1] range, with most parameters oscillating (η_i near 0). As batch size grows, something striking happens: the distribution goes *bimodal*. By bs=15000, almost all parameters have η_i ≈ 1 — they are each tracing near-monotonic paths. By bs=60000 it is a near-perfect spike at η_i = 1. Yet these are the batch sizes that barely learn anything. Every parameter commits fully to a direction; collectively, that direction is wrong. The small-batch regime — messy as it looks — has parameters actively exploring and correcting, which is exactly what learning requires.

---

##### Dynamic View: η1 is the Mirror of Learning

![Training curves](Jupyter%20Exploration/figs/5d_training_curves.png)

Watching η1 evolve over training is clarifying. For bs=60 (blue), loss collapses fast and η1 plummets from 1.0 to 0.10 over the full run — the two curves are nearly mirror images of each other. For bs=60000 (red), loss barely moves and η1 barely moves either, hovering near 0.93 throughout. With all eight batch sizes plotted, the spread makes the pattern even starker: every curve that learns aggressively (steep loss drop) pays for it with a steep η1 decay, and every curve that stays high in η1 barely moves in loss. The large-batch network is not "efficiently converging"; it is *efficiently going nowhere*. What looks like high efficiency in the final-scalar view is revealed here as stasis. The cumulative η1 decaying over time in the small-batch case is not a failure signal — it is the *signature of learning*.

---

##### TOPSIS: A Principled Single-Number Ranking

![TOPSIS distance to ideal](Jupyter%20Exploration/figs/5e_topsis_ranking.png)

With two competing objectives — minimize loss, maximize η1 — choosing the best batch size requires a joint ranking. TOPSIS normalizes both axes to [0, 1] and computes Euclidean distance to the ideal point (loss = 0, η1 = 1). The result is unambiguous: **bs=6000 (10% of the dataset) wins**, with a distance of ~0.19. The Pareto scatter (right panel) shows why geometrically: bs=60 sits near the bottom (best loss but η1 has decayed all the way to 0.10 after 10 epochs of small-batch chaos), large batches cluster at top-right (high η1, high loss), and bs=6000 uniquely achieves a middle position close enough to the ideal star on both axes simultaneously. This is the kind of ranking that would be invisible to any single-metric evaluation.

---

##### The Joint Landscape: Batch Fraction × Learning Rate

![Grid heatmaps](Jupyter%20Exploration/figs/8a_grid_heatmaps.png)

The 1-D sweep fixes learning rate. The 2-D grid relaxes that, sweeping six batch sizes against six learning rates (36 trials total). The four panels make the joint landscape legible at a glance.

- **η1 (top-left):** Increases left-to-right (bigger batch) and bottom-to-top (smaller LR). High LR + small batch = most chaotic motion. Low LR + large batch = most locally orderly.
- **Test loss (top-right):** The best cells are upper-right — high LR, small-to-medium batch. The loss landscape is nearly orthogonal to η1.
- **Qualified efficiency (bottom-left):** This is where the framework earns its keep. The brightest cells trace a **diagonal ridge** running from small-batch/high-LR toward large-batch/low-LR. This is the linear scaling rule — but it does not emerge cleanly from the loss heatmap, where "small batch + high LR" smears into a soft gradient with no visible structure. The efficiency metric crystallizes it: it simultaneously penalizes large-batch/low-LR configs (high η1, no learning) *and* small-batch/high-LR configs (good learning, chaotic motion), leaving only the co-scaled diagonal as the bright band. The loss heatmap tells you one objective is better in one corner; the efficiency heatmap tells you the *shape of the relationship between hyperparameters*.
- **TOPSIS (bottom-right, Purples_r):** The darkest cells — closest to the joint ideal — sit in the small-to-medium batch columns at high LR (top rows), with the optimum around **bs=600–3000, LR=3e-2**. The joint optimum is not a corner; it is an interior ridge that no single-axis sweep would reliably find.

This heatmap is arguably the most striking visualization the framework produces. The metrics do not just describe what happened — they make the *structure* of the hyperparameter landscape legible in a single image.

---

##### The Efficiency Ridge: LR Scales with Batch Size

![Efficiency ridge](Jupyter%20Exploration/figs/8b_efficiency_ridge.png)

For each batch size, the best-achievable qualified efficiency (across all LRs tested) traces a ridge. Two things stand out. First, the ridge is remarkably flat from bs=600 to bs=30000 — a wide plateau where the right LR yields ~0.67–0.70 qualified efficiency regardless of batch size. Second, the optimal LR jumps early and then locks in: bs=60 is best served by a low LR, but from bs=600 onward the optimal LR is consistently 3e-2 — larger batches all want the same aggressive learning rate. This is the **linear scaling rule made visible through a framework that has nothing to do with learning rate theory** — it emerges from the geometry of parameter-space motion alone. The left panel (one curve per LR) shows the mechanism: LR curves cross each other as batch size grows, with high-LR curves peaking at larger batches and low-LR curves peaking at smaller ones.

---

##### Summary

The framework surfaces structure that standard loss/accuracy reporting obscures entirely:

| Observation | What detected it |
| --- | --- |
| Small batches learn better but move chaotically | η1 + qualified efficiency together |
| Large batches are locally orderly but collectively incoherent | η2 gap |
| Per-parameter efficiency is bimodal, not uniform | η_i distribution histograms |
| η1 decay *is* the signature of learning | Training curve dynamics |
| The joint optimum is an interior point, not a corner | TOPSIS on the 2-D grid |
| Optimal LR scales with batch size | Efficiency ridge |

The qualified efficiency metric — η1 weighted by fractional test-loss improvement — is the key conceptual contribution of this sweep. It is the first scalar that simultaneously rewards directed motion *and* actual learning, penalizing both oscillation and stasis. The TOPSIS ranking and 2-D heatmap together show that this metric, combined with the trajectory geometry, makes the hyperparameter landscape navigable in a way that loss curves alone do not.
