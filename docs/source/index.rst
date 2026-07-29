Machine Learning Force Fields Data Collection
==========================================================

Project overview
----------------------------------------------------------

Machine learning force fields (MLFFs) can accelerate atomic simulations by
serving as computationally efficient surrogates for density functional theory
(DFT). Their usefulness, however, depends on whether they remain reliable when
atomic structures move beyond familiar training configurations.

This project asks: **When an MLFF is deliberately pushed away from equilibrium, does relaxation
return the material to its original structure?**

The study applies controlled adversarial perturbations to 20 two-dimensional
Materials Project structures and 20 Li-C-O configurations. Recovery is
evaluated across CHGNet, MACE, MACE-MH, UMA, and MTP using forces, atomic
displacements, relaxation convergence, symmetry, and structural topology.

FGSM, I-FGSM, and PGD are compared with non-adversarial contour exploration.
The experiments also examine perturbation magnitude, numerical precision,
random seeds, and supercells.

What we found
-------------

Perturbations approaching 10% of the minimum lattice parameter increasingly
prevent structures from recovering their pre-attack configurations. At this
magnitude, the largest post-relaxation force changes observed for the
LiCOHPF-trained MACE and MTP models reach approximately five orders of
magnitude above those observed for the pretrained general-purpose models.

This work provides a reproducible framework for testing MLFF reliability
inside an iterative physical workflow, where every force prediction affects
the next atomic configuration.

`View the project repository
<https://github.com/declanwhchan/mlff_attack_data_collection>`_

.. toctree::
   :maxdepth: 2
   :hidden:

   Motivation <motivation>
   Experiment Design <experimental_design>
   Getting Started <getting_started>
   Metrics <metrics>
   Outputs <outputs>
   Plot Methods <plot_methods>
   Results <results_index>
   Findings <findings>
   References <references>
