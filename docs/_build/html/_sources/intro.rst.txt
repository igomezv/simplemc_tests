Requirements
=============

This code runs both in Python 2x and 3x. However, we highly recommend Python 3x.

You need the following scientific modules:

.. code-block:: bash
   
   sudo pip install numpy matplotlib scipy nestle 


In addition, MCEvidence is necessary to estimate the bayesian evidence in the Metropolis-Hastings sampler:

.. code-block:: bash
   
   pip install git+https://github.com/yabebalFantaye/MCEvidence

For use Artificial Neural Networks with multinest and ellipsoidal sampling (as in pyBAMBI), you need to install:

.. code-block:: bash
   
   pip install tensorflow keras


If you want to use the full options to plot:

.. code-block:: bash
   
   pip install Pillow mpl_toolkits corner getdist

.. note:: All in one copy-paste line: 

   .. code-block:: bash
   
      pip install numpy matplotlib scipy nestle tensorflow keras Pillow mpl_toolkits corner getdist git+https://github.com/yabebalFantaye/MCEvidence



Quick Start
============

Create an *ini file* :

.. code-block:: none

   [DEFAULT]

   prefact = py ;[py, pre] only for Metropolis-Hastings

   chainsdir=chains 

   priortype = u    

   [custom]
 
   model = LCDM 
 
   datasets = SN+HD+BBAO 
 
   sampler = mnest 

   nsamp = 5000
   
   nlivepoints = 800 
   
   accuracy= 0.6 
   
   skip = 0 

   plotter= getdist 

.. note::

   Considerations:
  
   * *prefact* and *nsamp* are only for Metropolis-Hastings.

   * *nlivepoints* and *accuracy* are only for nested sampling.

   * *sampler* options are:
   
      * mh : Metropolis-Hastings.
      * snest : Single Nested Sampling (Ellipsoidal Nested Sampling)
      * mnest : MULTINEST
      * sbambi : snest + Artificial Neural Network
      * bambi : mnest + ANN

   * *plotter* can be getdist, corner or cosmich.

   * *skip* is burnin. 
  
   * For *priortype* u is uniform prior and g gaussian prior. At this time, only nested sampling accept both of them.
   
   * *chainsdir* is the directory where the chains in a text file and the plots will be saved.

Then you can run in the *SuperMC* directory:

.. code-block:: bash
   
   python Run/driver.py file.ini

* See the `plots <plotters.html>`_ .


General flow
=============

.. figure:: /img/SuperMCDiagram.png

