How the Plots Are Generated
===========================

The figures are generated from run-level summary tables produced after the
simulation outputs have been validated and combined. Failed or unavailable
rows are reported separately and are not silently interpreted as successful
measurements.

Perturbation-response curves
----------------------------

For each attack, model, and normalized epsilon, run-level measurements are
grouped across materials. The plotted line is the median, which summarizes the
typical response while reducing sensitivity to catastrophic outliers. Shaded
bands show the reported confidence interval or interquartile interval,
depending on the figure. Epsilon is displayed on a logarithmic horizontal
axis. Delta-force panels also use a logarithmic vertical scale because force
changes span several orders of magnitude.

Before-and-after recovery plots
-------------------------------

Each matched run contributes an initial measurement and a final
post-relaxation measurement. The two values are connected with a dotted line,
allowing the direction and magnitude of change to be read directly. Materials
are ordered using their calculated metric ranks rather than their numerical
identifiers.

Ranking and violin plots
------------------------

Run-level measurements are grouped by model. A violin represents the
distribution, the internal bar represents the interquartile range, and the
central point represents the median. Models are ordered by the actual median
metric value, with lower values ranked first when lower indicates better
recovery. Delta-force ranking plots use a logarithmic metric axis so that one
extreme model does not compress the remaining distributions.

Random-seed figures
-------------------

Equivalent rows from seeds 42 through 46 are aligned by material, model,
attack, epsilon, precision, and other experimental parameters. Thin lines show
individual seeds, while the thick line shows the cross-seed median and the
shaded region shows the interquartile range. Available matched rows are
plotted even when the full experiment table is incomplete; sample counts
should therefore be reported with the figure.

Precision-comparison plots
--------------------------

Float32 and float64 results are matched using the same model, structure,
attack, and attack parameters. Each point compares the two precisions for one
matched run. The dashed identity line represents perfect agreement, marginal
histograms show the distributions, and the reported coefficient summarizes
the strength of the relationship. Relaxation-step values must remain within
the configured range of 0 to 300; larger values indicate an accumulated or
stale trajectory and must be regenerated rather than clipped silently.

Immediate-response topology plots
---------------------------------

Topology is measured immediately after perturbation and before the second
relaxation. Curves summarize neighbour-edge Jaccard distance, radial
distribution function distance, and coordination-number change. These
measurements test whether the attack changes local connectivity before the
optimizer has an opportunity to recover the structure.

Post-relaxation topology plots
------------------------------

The same topology metrics are recalculated after relaxation and compared with
the pre-attack relaxed reference. A small immediate change followed by a small
final change indicates recovery. A persistent final change indicates that the
relaxation ended in a structurally different configuration.

Contour exploration plots
-------------------------

Contour trajectories are grouped by model, material, and contour parameter.
Their force, displacement, and topology responses provide a non-adversarial
reference against which the gradient-based attacks can be compared.

Supercell plots
---------------

Supercell analyses repeat the structural comparison after expanding the
periodic cell. Metrics are calculated using consistent periodic-neighbour and
minimum-image conventions. These plots test whether observed failure modes
remain visible when the represented system size changes.
