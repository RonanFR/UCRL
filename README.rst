Exploration-Exploitation in Reinforcement Learning
**************************************************
This library contains several algorithms for exploration-exploitation in RL
In particular we have implemented UCRL2, UCRL2B, SCAL, KL-UCRL2, TSDE, BKIA/OLP.

Note that this is a research project and by definition is unstable. Please write to us if you find something not correct or strange.

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

The suggested way of using this on MAC OS X is through conda

.. code:: shell

    conda install llvm gcc libgcc


See https://github.com/daler/pybedtools/issues/259 if you have troubles with compilation on MAC OS X

Testing
=======

We are using `pytest <http://doc.pytest.org>`_ for tests. You can run them via:

.. code:: shell

	  pytest
