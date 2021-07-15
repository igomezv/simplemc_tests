**Models**
==========

BaseCosmology class
---------------------

This class is the general structure of the models. It have the following methods:

* **setCurvature(self,R)**

*  **setrd(self,rd)**

*  **setVaryPrefactor(self,T=True)**


*  **setPrefactor(self,p)**

*  **prefactor(self)**
    
    Returns: self.c/(self.rd*self.h*100)

* **freeParameters(self)**

* **printFreeParameters(self)**

* **printParameters(self,params)**
  
   .. code-block:: python

     for p in params:
        print(p.name,'=',p.value,'+/-',p.error)

* **updateParams(self,pars)**


* **RHSquared_a(self,a)**
   This is relative hsquared as a function of *a* scale factor,   i.e. H(z)^2/H(z=0)^2.

        
* **Hinv_z(self,z)**

   Returns: 1/sqrt(self.RHSquared_a(1.0/(1+z)))


* **DistIntegrand_a(self,a)**
        return 1/sqrt(self.RHSquared_a(a))/a**2


* **Da_z(self,z)**

* **DaOverrd(self,z)**   
   D_a / rd

* **HIOverrd(self,z)**
   H^{-1} / rd
   
* **DVOverrd(self,z)**
    Dv / rd
   

    
* **distance_modulus(self,z):**

   Distance modulus.

   Note that our Da_z is comoving, so we're only multilpyting with a single (1+z) factor.



* **GrowthIntegrand_a(self,a):**    

   Returns the growth factor as a function of redshift

* **growth(self,z):**
   Equation 7.80 from Doddie
   
.. note::

   Base Cosmology class doesn't know about your parameterization of the equation of state or densities or anything. However, it does know about Hubble's constant at z=0 OR the prefactor c/(H0*rd) which should be fit for in the case of "rd agnostic" fits. That is why you should let it declare those parameterd based on its settings However, to get the angular diameter distance you need to pass it its Curvature parameter (Omega k basically), so you need to update it. 
  

models.LCDMCosmology
---------------------

.. note::

   This is LCDM cosmology. It is used as a base class for most other cosmologies, mostly because it treats Neutrinos and Radiation hassle.



The models modules need of the following modules:

models.ParamDefs
------------------

.. automodule:: models.ParamDefs
    :members:
    :undoc-members:
    :show-inheritance:

models.GenericParamDefs
------------------------

.. automodule:: models.GenericParamDefs
    :members:
    :undoc-members:
    :show-inheritance:

models.CosmoApprox
--------------------

.. automodule:: models.CosmoApprox
    :members:
    :undoc-members:
    :show-inheritance:


models.NuDensity
-----------------

This module calculates the predictions for the evolution
of neutrino energy densities.


models.RadiationAndNeutrinos
-----------------------------

This is a class that provides relevant support for treating radiation and neutrinos
Much of this functionality was in BasicCosmo, but it became clutterish there.



