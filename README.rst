Exploration-Exploitation in Reinforcement Learning
**************************************************
This library contains several algorithms based on the Optimism in Face of Uncertainty (OFU) principle both for MDPs and SMDPs.
In particular we have implemented
- UCRL [1]
- SMDP-UCRL and Free-Parameter SMDP-UCRL [2]
- SCAL [3]
All the implementations uses both Hoeffding's or Bernstein's confidence intervals.

References:
[1] Jaksch, Ortner, and Auer. Near-optimal regret bounds for reinforcement learning. Journal of Machine Learning Research, 11:1563–1600, 2010.
[2] Fruit, Pirotta, Lazaric, Brunskill. Regret Minimization in MDPs with Options without Prior Knowledge. NIPS 2017
[3] Fruit, Pirotta, Lazaric, Ortner. Efficient Bias-Span-Constrained Exploration-Exploitation in Reinforcement Learning. arXiv:1802.04020


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
For the ICML paper



What's new
==========
For a complete list of changes you can check `relative link`_.

.. _relative link: UCRL/__init__.py

- 2018-02-08: Release (v0.28.dev0)
    Contains SCAL and SCOPT (see Fruit, Pirotta, Lazaric, Ortner. Efficient Bias-Span-Constrained Exploration-Exploitation in Reinforcement Learning. arXiv:1802.04020)
- 2017-11-21: Release (v0.19.dev0)
    Added Span Constrained EVI
- 2017-11-16: Initial release (v0.15.dev0)
    Contains UCRL, SMDP-UCRL, Free-Parameter SMDP-UCRL (see Fruit, Pirotta, Lazaric, Brunskill. Regret Minimization in MDPs with Options without Prior Knowledge. NIPS 2017)
