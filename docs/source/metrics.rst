.. _metrics:

Metrics and Equations
=====================

This page defines the attack updates and reported metrics. In every structural
comparison, ``initial`` means the pre-attack relaxed reference and ``final``
means either the immediately perturbed or post-relaxation structure,
depending on the evaluation stage.

.. _metric-attack-methods:

Attack methods
------------------

FGSM
~~~~

FGSM applies one sign-gradient update to the atomic coordinates:

.. math::

   \mathbf{R}_{\mathrm{adv}}
   =
   \mathbf{R}_{0}
   +
   \epsilon\,
   \operatorname{sign}
   \left(
      \nabla_{\mathbf{R}}J_\theta(\mathbf{R}_{0})
   \right).

Here, :math:`J_\theta` is the configured attack objective and the model
parameters :math:`\theta` remain fixed.

:ref:`FGSM <method-ref-fgsm>`

I-FGSM
~~~~~~

I-FGSM applies repeated sign-gradient updates:

.. math::

   \mathbf{R}_{t+1}
   =
   \Pi_{\epsilon}
   \left[
      \mathbf{R}_{t}
      +
      \alpha\,
      \operatorname{sign}
      \left(
         \nabla_{\mathbf{R}}J_\theta(\mathbf{R}_{t})
      \right)
   \right].

:math:`\alpha` is the step size and :math:`\Pi_\epsilon` keeps the coordinates
inside the permitted perturbation region.

:ref:`I-FGSM <method-ref-ifgsm>`

PGD
~~~

PGD applies iterative updates projected into the allowed region:

.. math::

   \mathbf{R}_{t+1}
   =
   \Pi_{\mathcal{B}_\epsilon(\mathbf{R}_0)}
   \left[
      \mathbf{R}_{t}
      +
      \alpha\,
      \operatorname{sign}
      \left(
         \nabla_{\mathbf{R}}J_\theta(\mathbf{R}_{t})
      \right)
   \right].

PGD can use a randomized starting point inside the allowed region.

:ref:`PGD <method-ref-pgd>`

Normalized epsilon
------------------

The attack budget is normalized by the shortest lattice-vector length:

.. math::

   \epsilon_{\%}
   =
   100
   \frac{
      \epsilon
   }{
      \min
      \left(
         \lVert\mathbf{a}\rVert_2,
         \lVert\mathbf{b}\rVert_2,
         \lVert\mathbf{c}\rVert_2
      \right)
   }.

This is the nominal attack budget, not the final post-relaxation displacement.

:ref:`NumPy norm <method-ref-numpy-norm>`

Relaxation convergence
----------------------

A relaxation converges when the largest atomic force norm satisfies:

.. math::

   \max_i
   \left\lVert
      \mathbf{F}_i
   \right\rVert_2
   \leq
   0.01\ \mathrm{eV\,\AA^{-1}}.

The optimizer is limited to 300 steps.

:ref:`ASE optimization <method-ref-ase-optimize>`

Displacement
------------

Atomic displacement is the magnitude of the final-minus-initial position:

.. math::

   d_i
   =
   \left\lVert
      \mathbf{r}_{i,\mathrm{final}}
      -
      \mathbf{r}_{i,\mathrm{initial}}
   \right\rVert_2.

The pipeline summarizes these per-atom values using the configured mean,
median, or maximum.

:ref:`NumPy norm <method-ref-numpy-norm>`

Delta force
-----------

Delta force is the magnitude of the final-minus-initial force:

.. math::

   \Delta F_i
   =
   \left\lVert
      \mathbf{F}_{i,\mathrm{final}}
      -
      \mathbf{F}_{i,\mathrm{initial}}
   \right\rVert_2.

This compares two predictions from the same model. It is not a DFT force error.

:ref:`NumPy norm <method-ref-numpy-norm>`

Delta-force angle
-----------------

The force-angle change is:

.. math::

   \theta_i
   =
   \cos^{-1}
   \left(
      \frac{
         \mathbf{F}_{i,\mathrm{initial}}
         \cdot
         \mathbf{F}_{i,\mathrm{final}}
      }{
         \lVert\mathbf{F}_{i,\mathrm{initial}}\rVert_2
         \lVert\mathbf{F}_{i,\mathrm{final}}\rVert_2
      }
   \right).

The implementation clips the cosine to :math:`[-1,1]` and handles zero-force
cases separately.

:ref:`NumPy norm <method-ref-numpy-norm>`

Relaxation steps
----------------

The recorded step count is limited to 300:

.. math::

   n_{\mathrm{reported}}
   =
   \min
   \left(
      n_{\mathrm{optimizer}},
      300
   \right).

A value of 300 means the configured maximum was reached.

:ref:`ASE optimization <method-ref-ase-optimize>`

Neighbour-set Jaccard distance
------------------------------

Let :math:`E_{\mathrm{initial}}` and :math:`E_{\mathrm{final}}` be the
neighbour-edge sets. Their Jaccard distance is:

.. math::

   d_J
   =
   1
   -
   \frac{
      \left|
         E_{\mathrm{initial}}
         \cap
         E_{\mathrm{final}}
      \right|
   }{
      \left|
         E_{\mathrm{initial}}
         \cup
         E_{\mathrm{final}}
      \right|
   }.

Zero means identical connectivity; one means that no neighbour edges are
shared.

:ref:`SciPy Jaccard <method-ref-jaccard>`

Coordination change
-------------------

For coordination number :math:`z_i`, the per-atom coordination change is:

.. math::

   \Delta z_i
   =
   \left|
      z_{i,\mathrm{final}}
      -
      z_{i,\mathrm{initial}}
   \right|.

The pipeline reports the configured mean or maximum across atoms.

:ref:`Coordination <method-ref-coordination>`

RDF L1 distance
---------------

For initial and final radial distribution functions evaluated on the same
bins, the RDF L1 distance is:

.. math::

   d_{\mathrm{RDF}}
   =
   \sum_{k=1}^{K}
   \left|
      g_{\mathrm{final}}(r_k)
      -
      g_{\mathrm{initial}}(r_k)
   \right|
   \Delta r.

Zero means the two RDF curves are identical. The value depends on the RDF
cutoff, binning, and normalization.

The RDF is generated using ``ase.geometry.rdf.get_rdf``.

:ref:`ASE RDF <method-ref-rdf>`


Symmetry-operation retention
----------------------------

The retained fraction of initial symmetry operations is:

.. math::

   f_{\mathrm{retained}}
   =
   \frac{
      \left|
         S_{\mathrm{initial}}
         \cap
         S_{\mathrm{final}}
      \right|
   }{
      \left|
         S_{\mathrm{initial}}
      \right|
   }.

The result depends on the symmetry and operation-matching tolerances.

:ref:`Symmetry <method-ref-symmetry>`

Unique-site change
------------------

The absolute change in the number of crystallographically unique sites is:

.. math::

   \Delta n_{\mathrm{unique}}
   =
   \left|
      n_{\mathrm{unique,final}}
      -
      n_{\mathrm{unique,initial}}
   \right|.

The result depends on the symmetry tolerance.

:ref:`Symmetry <method-ref-symmetry>`

Cross-seed variability
----------------------

The random-seed shaded region uses the interquartile range:

.. math::

   \operatorname{IQR}
   =
   Q_{0.75}
   -
   Q_{0.25}.

Thin lines represent individual available seeds and thick lines represent the
available-seed median.

:ref:`NumPy quantile <method-ref-numpy-quantile>`
