Findings and Conclusions
===========================

Primary observation
-------------------

Recovery becomes substantially less reliable as the nominal perturbation
budget approaches 5% of the minimum lattice-vector length. At this scale,
many structures no longer return to configurations resembling their
pre-attack relaxed references.

Force instability
-----------------

The largest observed post-relaxation force changes for the LiCOHPF-trained
MACE and MTP models are approximately five orders of magnitude larger than
those observed for the pretrained general-purpose models at comparable
perturbation magnitudes.

This result should be interpreted as an observed benchmark outcome rather
than a universal ranking of the architectures. Training data, calculator
settings, chemistry, convergence behaviour, and a small number of extreme
runs may all contribute to the difference.

Why the result matters
----------------------

A model can produce predictions quickly and still guide an optimizer toward
an unreliable configuration. Conventional held-out errors do not fully
describe this sequential failure mode because relaxation repeatedly feeds
model predictions back into the next structural update.

The experiment therefore evaluates robustness at the workflow level rather
than only at the single-prediction level.

Limitations
-----------

The conclusions should be presented with the following qualifications:

* Nominal epsilon is an attack budget normalized by lattice size; it is not
  identical to final post-relaxation displacement.
* A run reaching 300 steps reached the configured limit and should not be
  described as converged without an independent convergence check.
* Extremely large force changes require inspection for numerical instability,
  invalid geometries, calculator failure, or stale trajectory data.
* Missing rows change sample counts and must be disclosed in aggregated plots.
* Random-seed comparisons quantify stochastic pipeline variability, not
  uncertainty from independently retraining every pretrained model.
* Comparisons between custom-trained and broadly pretrained models combine
  architecture and training-distribution effects.

Future work
-----------

Future studies should compare adversarial results with direct DFT
recalculations, expand the chemistry and defect coverage, quantify uncertainty,
test alternative optimizers and convergence criteria, and determine whether
adversarially selected configurations improve active-learning or retraining
workflows.
