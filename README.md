# simpleMC

Personal copy of the `SimpleMC` cosmological parameter estimation code. The original source, where the latest stable version is located, is in the repository of Dr. José Alberto Vázquez: https://github.com/ja-vazquez/SimpleMC 

Whithin the simplemc directory (where is the setup.py file) you can install in your computer simplemc via: `pip install -e .`

     $ git clone https://github.com/igomezv/simplemc_tests
     $ cd simplemc_tests
     $ pip3 install -e .

then you can delete the cloned repo because you must have locally installed `simplemc`. 

Other way to install `simplemc` is:

     $ pip3 install -e git+https://github.com/igomezv/simplemc_tests


In the `requirements.txt` file there are the basic libraries to run `simplemc`, but some functions such as plots or neural networks can be unavailable. For full requirements check `requirements_full.txt`.

In the `docs` directory are all the files used by the `sphinx` library to build the website with the `SimpleMC` temporary documentation: https://igomezv.github.io/SimpleMC/ 

In this current repository I do some personal tests and experiments before contributing to the original source. 

Any question or suggestion please contact me.

TO DO:

- Prints of Pantheon likelihood.
- Add warnings in bad combinations of RC and Simple likelihoods with some models.
- Unify pybambi and nerualike.
- Neural networks methods with dynesty multiprocessing fails. 
