===
API
===

This page details the methods and classes provided by the ``SimpleMC``.

Top level interface
********************

.. automodule:: simplemc.DriverMC
    :members:
    :show-inheritance:

.. automodule:: simplemc.CosmoCalc
    :members:
    :show-inheritance:

.. automodule:: simplemc.PostProcessing
    :members:
    :show-inheritance:

Likelihood
***********

.. automodule:: simplemc.likelihoods.BaseLikelihood
    :members:
    :show-inheritance:

.. automodule:: simplemc.likelihoods.CompressedSNLikelihood
    :members:
    :show-inheritance:
  
Cosmo
******

.. automodule:: simplemc.cosmo.Parameter
    :members:
    :show-inheritance:

.. automodule:: simplemc.cosmo.BaseCosmology
    :members:
    :show-inheritance:

.. automodule:: simplemc.cosmo.RadiationAndNeutrinos
    :members:
    :show-inheritance:

.. automodule:: simplemc.cosmo.NuDensity
    :members:
    :show-inheritance:

.. automodule:: simplemc.cosmo.Derivedparam
    :members:
    :show-inheritance:

Models
******
.. automodule:: simplemc.models.LCDMCosmology	
    :members:
    :show-inheritance:

.. automodule:: simplemc.models.owa0CDMCosmology
    :members:
    :show-inheritance:

.. automodule:: simplemc.models.SimpleModel	
    :members:
    :show-inheritance:



Analyzers
**********

.. automodule:: simplemc.analyzers.MCMCAnalyzer
    :members:
    :show-inheritance:

.. automodule:: simplemc.analyzers.MaxLikeAnalyzer
    :members:
    :show-inheritance:

.. automodule:: simplemc.analyzers.GA_deap
    :members:
    :show-inheritance:

.. automodule:: simplemc.analyzers.SimpleGenetic
    :members:
    :show-inheritance:

.. automodule:: simplemc.analyzers.Population
    :members:
    :show-inheritance:

.. automodule:: simplemc.analyzers.neuralike.NeuralManager
    :members:
    :show-inheritance:

We use several nested sampling algorithms included in ``dynesty`` library. We slightly modify ``dynesty`` to save the strings in the text file and also to incorporate neural networks. However, its `documentation <https://dynesty.readthedocs.io/en/latest/>`_  is very complete, clear and didactic, so we recommend that you visit it.

Plots
*****
.. automodule:: simplemc.plots.Simple_Plots
    :members:
    :show-inheritance: 
