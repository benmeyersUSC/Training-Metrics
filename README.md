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

The sweep trains a small feedforward network (784 → 128 → ReLU → 10) on the MNIST digit classification dataset across eight batch sizes (from 0.1% to 
100% of the 60k training set) measuring η1, η2, loss, accuracy, and **qualified efficiency** (η1 × fractional test-loss improvement) at each configuration.

---

##### The Surface-Level View: Trade-offs

![Trajectory metrics vs batch fraction](Jupyter%20Exploration/figs/5a_final_metrics_vs_batch_fraction.png)

The top row of this plot tells a familiar story. Batch learning is known to trade coherence of movement for quicker
descent. When a gradient signal updates the network's weights, it matters if it carries the lessons of 100 examples vs
1000 examples vs 1000000. Ideally, every update would be motivated by the summed loss gradient of the entire dataset.
This is an expensive computation, though. It has been found that using smaller batches for updates can give you a messier, but
quicker, descent as you hope that the batch of samples you've backpropagated carries representative information of the 
whole dataset. 

We see the dynamics of batching clearly in the first row: as batch size grows, each update is more directionally informed, 
giving higher L1 and L2 efficiency. With this efficiency of movement, though, comes trepidation; the juiced gradient signal
of a large batch has lots of cancelling pressure. Yes, this cancelling represents the informational diversity of the data, 
but it slows descent. This is why higher batch sizes, for their efficiency of movement, move less down the loss landscape. 

This trade-off spectrum for batch size suggests a better metric: **qualified efficiency (Q)**. Can we scale our measurement of 
movement efficiency by how productive that efficient moveemnt actually was? This is precisely what **Q** measures. 
On the bottom right plot, we see **Q** against batch size, giving us a very powerful read on which configuration
gets both the big-batch benefits of efficient updates (gross movement approaches net) *and* the small-batch benefits
of quick gradient descent. The peak at 6000 = 10% tells us that for a static batch size over the course of a 10-epoch
training run of this architecture on MNIST, updating on 10% of the dataset at a time yields efficient, productive motion.

The Pareto scatter in the bottom middle plot also gives us a geometric intuition for the approach to an optimally efficient
and productive batch size. The best possible hyperparameter configuration for a neural network training run would be one
with **Final Loss = 0** and **η1 = 1.0** (coordinates [0,1], top left corner). The curve shown by these points represents, again,
the **Q** trade-off presented by batch size. This Pareto metric is later normalized and improved to look at **ΔLoss%**.

---

##### η1 vs η2: Every Parameter Marches Alone

![η1 vs η2 gap](Jupyter%20Exploration/figs/5b_eta1_eta2_gap.png)

η2, the measurement of geodesic-ness of global network movement through parameter space is an inherently harder metric
to maximize, especially given how backpropagation is calculated. Regardless of your hyperparameters, the derivatives composing
the network gradient are all partial, local, and dependent on their neighbors and descendents. η1 is still a global summation
of the efficiency of the parameters of a network, but it sums *individual coherence* of parameter movement. It is an 
average of the internal, local coherence/efficiency of movement of all parameters. η2, on the other hand, is a measure of
how coherent the *collective* movement of all parameters is. But parameters are nudged according to *individual* interests. 
Comparing η2 and η1 is not quite fair, at least on the same scale. It is, however, helpful and useful to see and know why!

Importantly, though, regardless of η2's elusiveness for partial-derivative-driven optimization, it is still a useful
metric itself as a comparator between hyperparameter or loss regimes for networks.

---

##### Inside the Network: The Bimodal Efficiency Split

![Per-parameter η_i distributions](Jupyter%20Exploration/figs/5c_per_param_eta_distributions.png)

η1 is an average of individual parameter efficiency. These histograms unpack the mean into its component points. For each
batch size, we've plotted the distribution of efficiency for all model parameters. This gives us a sense of the distribution
guiding the single mean η1. As batch size increases, we can observe the efficiency mass migrating rightward. This is, again,
a direct consequence of the enrichment of large-batch gradient signals as compared to small-batch. By the full batch run, 
nearly all of the parameters are, individually, moving in a uniform direction. The full dataset prescribes a specific
route that a small-batch simply cannot replicate with fidelity. 

An interesting question to ponder in response to this figure is: if parameter movement is so incoherent in small-batch
training runs, how does the model learn so quickly? Nearly all of the motion contains waste. But another way to think about
waste is *exploration*. The small batch updates are thrashing parameters around in a messy search; but this search will inevitably
find useful configurations and will retain them. This is learning. 

---

##### Dynamic View: η1 is the Mirror of Learning

![Training curves](Jupyter%20Exploration/figs/5d_training_curves.png)

Having previously plotted η1 and CE Loss for different regimes after the full training run, we wanted to take a look at
η1 and Loss over time for all regimes at once. We see a familiar loss curve on the left and, its shockingly similar-looking
cousin, the η1 plot. The steepest loss curves on the left come from the least efficient regimes, as seen on the right. What, 
only looking at η1, looks like efficient and ideal learning is revealed here as stasis. What seemed like chaotic thrashing
shows up here as rapidly time- and compute-efficient learning. 

Qualified efficiency, **Q**, attempts to unify the two valid priorities (movement efficiency and learning speed). But beyond
unification into one metric, we have to recognize that high-speed learning and highly coherent learning serve different purposes. 
When I want to train a network to be really useful, cheaply and quickly, I will use (or at least start with) small batch
training. There is no reason not to tumble down the loss landscape if I just want results. However, when we ask questions
about *what neural network learning is* and *when parameters move to the right place, what pulls them*, we see more value
in studying efficient parameter movement. Small-batch training is a clever hack to approximate the descent of perfect
full-batch training. Studying *what is being approximated* by small-batch training is vital to understanding the 
principles of learning as movement. 


---

##### TOPSIS: A Principled Single-Number Ranking

![TOPSIS distance to ideal](Jupyter%20Exploration/figs/5e_topsis_ranking.png)


Returning to qualified efficiency, **Q**, we can calculate the principled, normalized distance to optimality. We improve
the distance measure to a normalized TOPSIS value, representing the distance any given training run is from optimal:
**η1 = 1.0** and **ΔLoss = 100%**. With axes normalized to [0, 1], we can take the Euclidian distance of any point
to the optimal **(0,1)**. The rankings are the same as above, but now put in terms of a unified, standardized metric. 


---

#### The Joint Landscape: Batch Fraction × Learning Rate

After doing an initial scan of the framework's diagnostic capacity when tuning batch size, we run through an example of
testing hyperparameters on a grid of two axes: batch size and learning rate. This exploration, as with the batch size 
sweep, is not necessarily for practical application, but instead to get a read on the ability of the framework to 
explain well-known dynamics in neural network training. Practically, though, this kind of exploration to optimize an 
actual training regime may look like this. 

![Grid heatmaps](Jupyter%20Exploration/figs/8a_grid_heatmaps.png)

Testing batch size from 0.1% to 50% of the dataset size and learning rate from 1e-4 to 3e-2, we initially produced four
grids to measure performance of each configuration. 

- **η1 (top-left):** This is unqualified mean efficiency of parameter movement. Familiar dynamics can be seen: batch size
is clearly correlated with η1. Similarly, looking down any column we see that lower learning rate points to higher efficiency. The
most motion-efficient training regime has maximal batch size and minimal learning rate. Why? Firstly, when you train on
the full dataset each update, there is 0 noise in the gradient. The gradient you accumulate reflects the entirety of the
local loss landscape; in other words, full-batch gradients always point in exactly the correct direction of parameter movement. 
If the full-batch gradient points in the correct direction, why not use a learning rate of 100% Because the gradient computed
by backpropagation is local and typical NN loss landscapes are not linear. There is not one slope that globally defines the 
direction of descent; too large a learning rate will cause you to, even on full-batch gradients, overstep and correct. 
This is precisely why as learning rate approaches 0, efficiency approaches 1. As you lower learning rate, you zoom ever
in on the loss landscape, making it more locally linear. Full batch gradient with infinitesimal step size is maximally 
efficient movement. However, at this efficiency limit, the parameters are barely moving and the model is learning nothing!
The efficiency of movement, given by the full-gradient information and highly conservative step, come at a cost! This is why we have **Q**.
- **Test loss (top-right):** Here, we see the complementary triangle over the grid. Given that each trial was a fixed number
of epochs, this final test loss serves more as a rate of loss than an absolute value to take home. As the dynamics discussed
above suggested, the bottom right of the grid is where no learning happens. Too much was spent on efficiency and the model
moved nowhere. In this final big-batch column, we can see that as you increase the learning rate (committed step for this
informed gradient), you get more learning. However, while we see a bias to upper-left, the global minimum loss in this grid 
rides a diagonal ridge in the top left corner. 
- **Qualified efficiency, Q (bottom-left):** This **Q** grid rightly appears to be a kind of combination of the η1 and loss grids. Because
efficiency of movement and learning are competing interests both being maximized, we see a diagonal ridge through the middle
of the grid, reflecting compromise. It suggests that efficiency of movement and meaningfulness of movement are dually important
for good training. It also suggests a beautiful relationship between batch size and learning rate, demanding give and take. 
If you increase your learning rate, you had better have a more informed gradient; complementarily, if you are putting more
work into computing your gradient (more examples) then you had better not waste it on a small step. 
- **TOPSIS (bottom-right):** Identical dynamics as **Q** grid, but this time in terms of distance to optimum. 

---

##### The Efficiency Ridge: LR Scales with Batch Size

![Efficiency ridge](Jupyter%20Exploration/figs/8b_efficiency_ridge.png)

For each batch size, the best-achievable qualified efficiency (across all LRs tested) traces a ridge. 
Two things stand out. First, the ridge is remarkably flat from bs=600 to bs=15000 — a wide plateau where the
right LR yields ~0.67–0.70 qualified efficiency regardless of batch size. Second, the optimal LR jumps early and then 
locks in: bs=60 is best served by a low LR, but from bs=600 onward the optimal LR is consistently 3e-2 — larger batches
all want the same aggressive learning rate. This is the **linear scaling rule made visible through a framework that has
nothing to do with learning rate theory** — it emerges from the geometry of parameter-space motion alone. The left panel 
(one curve per LR) shows the mechanism: LR curves cross each other as batch size grows, with high-LR curves peaking at 
larger batches and low-LR curves decaying at smaller ones.

---

##### The Efficiency Landscape Shifts Under Training

![Epoch efficiency evolution](Jupyter%20Exploration/figs/9b_epoch_efficiency_evolution.png)

The 2-D grid sweeps above report *final-epoch* metrics. These ten panels show the full qualified-efficiency heatmap at each epoch snapshot, on a shared color scale. The story changes when you watch it evolve.

In early epochs, the brightest cells sit at LR=3e-2 (the top row) with small-to-medium batch sizes — the model is actively learning and any configuration that combines directed motion with actual loss reduction scores well. As training progresses, configurations with small batch sizes exhaust their gains first: their η1 has decayed and loss has plateaued in the current regime, so their qualified efficiency collapses. The bright region migrates rightward toward larger batch sizes. The LR=3e-2 row stays optimal throughout; only the optimal batch size changes.

The implication is direct: **a fixed hyperparameter configuration can only be optimal for a subset of epochs.** An adaptive schedule that tracks the moving bright cell should achieve higher cumulative qualified efficiency than any fixed baseline — and the η1 signal is a natural sensor for detecting when a configuration has been exhausted.

---

##### η1-Triggered Co-Scaling Scheduler

The per-epoch evolution raises a natural question: can we build a scheduler that *rides* the moving efficiency ridge rather than committing to a fixed position?

**The mechanism.** η1 decays monotonically as the model extracts information from its current (batch_size, LR) regime — each epoch the gradient directions become more consistent, reflecting diminishing returns, not increasing mastery. When the per-epoch drop Δη1 falls below a threshold, the regime is exhausted. That is the signal to advance.

**The protocol.** Step through discrete (batch_size, LR) pairs along the grid diagonal when Δη1 < 0.05. Start at (bs=6000, lr=3e-3) — the TOPSIS-optimal batch size from the 1-D sweep — and advance through (bs=15000, lr=1e-2) → (bs=30000, lr=3e-2) as η1 plateaus.

![η1-scheduled epoch heatmaps](Jupyter%20Exploration/figs/10_scheduled_epoch_heatmaps.png)

Each panel is the background grid qualified-efficiency heatmap for that epoch with a red box highlighting the scheduler's active cell. Around epoch 3 the scheduler advances (panels titled in firebrick with "↗ scaled"), stepping the red box one position up the diagonal. The box is consistently in the brighter half of the grid — unlike a naive small-batch start, which would sit in the darkest cells for the majority of the run.

![η1-scheduled vs fixed baselines](Jupyter%20Exploration/figs/10_scheduled_vs_baselines.png)

The left panel (per-epoch qualified efficiency) and right panel (cumulative mean) compare the scheduler against three fixed baselines.

- **Fixed bs=60 (blue):** Starts high (~0.70 at epoch 1) because the small-batch model is actively learning, but η1 collapses quickly as the model converges — by epoch 10, qual eff has decayed to ~0.10 and the cumulative mean is ~0.15.
- **Fixed bs=600 (orange):** A more moderate decay — starts at ~0.70 and falls to ~0.39 by epoch 10; cumulative mean ~0.50.
- **Fixed bs=6000 (green):** Starts very low (~0.28) because at epoch 1 the large-batch model hasn't yet converged enough to show strong fractional improvement, but climbs steadily as it settles into a productive basin, reaching ~0.73 by epoch 10; cumulative mean ~0.67.
- **η1-scheduled (yellow):** Starts at ~0.62 — immediately in the good region — peaks near ~0.74 around epoch 3, dips briefly at the step advance, then stabilizes around ~0.70 through the later epochs; cumulative mean ~0.67.

**The scheduler matches the best fixed baseline (bs=6000) on overall cumulative quality, but without the warm-up cost.** Fixed bs=6000 needs several epochs of low efficiency before it climbs into the productive range; the scheduler is already there from epoch 1. The advantage is front-loaded, which matters when training budgets are tight.

**What this demonstrates.** The η1 decay is a real signal — not noise, not an artifact of the network size or dataset. It identifies when a regime has been consumed, and stepping on that signal produces a trajectory that is competitive with the empirically best fixed configuration without knowing in advance which configuration that is.

---

##### Open Questions

The scheduler as implemented is a first sketch. Several questions remain open:

- **Path optimality.** The grid diagonal is one path through the landscape. The 9b evolution shows the optimal cell stays in the LR=3e-2 row throughout — a schedule that fixes LR at the peak and only scales batch size may outperform the diagonal.
- **Transition cost.** The brief per-epoch dip at the step advance (visible in the left panel) suggests switching configurations disrupts the accumulated trajectory. Resetting the G/D accumulators at each step would make η1 reflect *only* the current phase, potentially sharpening the plateau signal and smoothing transitions.
- **Threshold sensitivity.** The Δη1 threshold (0.05) is a free parameter. Too high: the scheduler advances prematurely, spending too little time in each cell. Too low: it fails to advance before efficiency has already collapsed.
- **Continuous gradient.** With enough epoch-snapshot data, one could fit η1(batch_size, lr, epoch) and compute ∂η1/∂bs and ∂η1/∂lr analytically — continuous gradient ascent through the landscape rather than discrete diagonal stepping.

---

##### Summary

The framework surfaces structure that standard loss/accuracy reporting obscures entirely:

| Observation | What detected it |
| --- | --- |
| Small batches learn better but move chaotically | η1 + qualified efficiency together |
| Large batches are locally orderly but collectively incoherent | η2 gap |
| Per-parameter efficiency is bimodal at intermediate batch sizes | η_i distribution histograms |
| η1 decay *is* the signature of learning | Training curve dynamics |
| The joint optimum is an interior point, not a corner | TOPSIS on the 2-D grid |
| Optimal LR scales with batch size | Efficiency ridge |
| The optimal configuration migrates over the course of training | Per-epoch heatmap evolution |
| η1 plateau is a valid sensor for hyperparameter advancement | η1-triggered scheduler |

The qualified efficiency metric — η1 weighted by fractional test-loss improvement — is the key conceptual contribution of this sweep. It is the first scalar that simultaneously rewards directed motion *and* actual learning, penalizing both oscillation and stasis. The TOPSIS ranking, 2-D heatmap, epoch evolution, and scheduler together show that this metric makes the hyperparameter landscape not just legible, but navigable.
