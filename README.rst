Exploration-Exploitation in Reinforcement Learning
**************************************************
This library contains several algorithms based on the Optimism in Face of Uncertainty (OFU) principle both for MDPs and SMDPs.
In particular we have implemented

- UCRL [1]

- SMDP-UCRL and Free-Parameter SMDP-UCRL [2]

- SCAL [3]

All the implementations uses both Hoeffding's or Bernstein's confidence intervals.

Note that this is a research project and by definition is unstable. Please write to us if you find something not correct or strange.

References:

`[1]`__ Jaksch, Ortner, and Auer. Near-optimal regret bounds for reinforcement learning. Journal of Machine Learning Research, 11:1563–1600, 2010. 

`[2]`__ Fruit, Pirotta, Lazaric, Brunskill. Regret Minimization in MDPs with Options without Prior Knowledge. NIPS 2017

`[3]`__ Fruit, Pirotta, Lazaric, Ortner. Efficient Bias-Span-Constrained Exploration-Exploitation in Reinforcement Learning. arXiv:1802.04020

__ http://www.jmlr.org/papers/volume11/jaksch10a/jaksch10a.pdf
__ https://papers.nips.cc/paper/6909-regret-minimization-in-mdps-with-options-without-prior-knowledge.pdf
__ https://arxiv.org/abs/1802.04020

Contributors
============

- Matteo Pirotta (INRIA Lille - SequeL Team)

- Ronan Fruit (INRIA Lille - SequeL Team)

Installation
============

You can perform a minimal install of the library with:

.. code:: shell

	git clone https://github.com/RonanFR/UCRL
	cd UCRL
	pip install -e .
	make
	

Testing
=======

We are using `pytest <http://doc.pytest.org>`_ for tests. You can run them via:

.. code:: shell

	  pytest
	  


.. _See What's New section below:

How to reproduce experiments
============================
In order to reproduce the results in [3] you can follow these instructions.
For SCAL, you can run the following command by changing the span constraint (5 and 10) and the seed (114364114, 679848179, 375341576, 340061651, 311346802). Results are averaged over 15 runs. You can change the number of repetitions by changing the parameter -r.

.. code:: shell

	  python ../example_resourcecollection.py --alg SCAL  --p_alpha 0.05 --r_alpha 0.05 --boundtype bernstein  -n 400000000 -r 3 --seed 114364114 --rep_offset 0 --path SCAL_KQ_c2 --span_constraint 10 --regret_steps 5000 --armor_collect_prob 0.01 
	  

For UCRL, you can run the same command by changing --alg to UCRL (the other parameters are the same and are ignored if not required by UCRL).


What's new
==========
For a complete list of changes you can check `UCRL/__init__.py`_.

.. _UCRL/__init__.py: UCRL/__init__.py

- 2018-02-08: Release (v0.28.dev0)
    Contains SCAL and SCOPT (see Fruit, Pirotta, Lazaric, Ortner. Efficient Bias-Span-Constrained Exploration-Exploitation in Reinforcement Learning. `arXiv:1802.04020`__)
- 2017-11-21: Release (v0.19.dev0)
    Added Span Constrained EVI
- 2017-11-16: Initial release (v0.15.dev0)
    Contains UCRL, SMDP-UCRL, Free-Parameter SMDP-UCRL (see Fruit, Pirotta, Lazaric, Brunskill. Regret Minimization in MDPs with Options without Prior Knowledge. NIPS 2017)

__ https://arxiv.org/abs/1802.04020

