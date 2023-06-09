Added analysis files to original hebbff models distributed at  https://github.com/dtyulman/hebbff: 

**Replication**
- replication.py: code to replicate key results from Tyulmankov et al. (2022)

**Certainty rating**
- uncertainty.py: code to generate `uncertainty/y.png` plot showing `a2` ouptut layer
distribution (before sigmoid non-linearity)
- uncertainty_clf.py: defines `Certainty`, a wrapper class for any StatefulBase model to
learn output distribution parameters and compute certainty ratings
- uncertainty_plots.py: plot certainty ratings for correct/incorrect predictions over `R`
and non-binarized output `a2` (`uncertainty/hist.png` and `uncertainty/scatter.png`)

**Information quantification**
- information.py: plot accuracy and encoded information over `R` (`information/HebbNet_Rtest.png`)

**State vs novel image recognition**
- state_images.py: plot accuracy (all, familiar, state) for binarized images from
Brady et al. (2008) (`state_figs/accuracy_R*.png`, `state_figs/inf.png`)
