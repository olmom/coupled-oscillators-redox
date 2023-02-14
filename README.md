<!-- badges: start -->
<!--<a href="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205155"><img src="https://img.shields.io/badge/Data-GSE205155-green.svg?style=plastic" alt="" /></a>-->
<!--[![](https://img.shields.io/badge/Data-10.1101/2022.06.03.494693-blue.svg)](https://doi.org/10.1101/2022.06.03.494693)-->
[![](https://img.shields.io/badge/Preprint-10.1101/2023.02.12.528191-yellow.svg)](https://www.biorxiv.org/content/10.1101/2023.02.12.528191v1)
<!--[![](https://img.shields.io/badge/Data-10.1101/2022.06.03.494693-blue.svg)](https://doi.org/10.1101/2022.06.03.494693)-->
 <!-- badges: end -->

# coupled-oscillators-redox

This repository contains the reproducible code for the generation of data, analysis and figures in the manuscript "Coupling allows robust circadian rhythms despite heterogeneity and noise" ([preprint](https://www.biorxiv.org/content/10.1101/2023.02.12.528191v1)). Note that the simulated data should be generated before.

To execute this code:

1. Clone this repository at a suitable location. This will place the code within a directory named **coupled-oscillators-redox**
2. Generate the simulated data using the *main.py* script. To do this, check the parameter combinations that are needed to generate each figure in the header of the scripts *fig1.py*, *fig2.py* ... and introduce those as arrays in the header of the *main.py* script. When you run the *main.py* script, the results of the stochastic simulations will be saved in a folder called **results** within the project 
   - Alternatively, all results are available on request and can be sent via *wetransfer* (note that this file will be in the order of ~90 GB)
<!--2. Download all the simulated data from [here](https://www.zenodo.org/) (under the `results` folder) (or alternatively generate all the simulated data using the *main.py* script)-->
3. Make sure you have Python 3.7+ and the following libraries installed: `numpy`, `scipy`, `matplotlib`, `statsmodels`, `pandas`, `seaborn`, `glob`
4. To reproduce the figures, the Python files can now be executed in order *fig1.py*, *fig2.py,* ... within the project 

The Python script used to generate results and figures rely on objects from the *poincare.py* script. A solver for stochastic differential equations based on the Euler Maruyama method (the `itoEuler` method) is also available in the same script as part of the `sdeIntegrator()` class.

<!--To reproduce the figures, the Python files can now be executed in order *fig1.py, fig2.py,* ... within the project.-->
