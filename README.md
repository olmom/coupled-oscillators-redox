<!-- badges: start -->
<!--<a href="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205155"><img src="https://img.shields.io/badge/Data-GSE205155-green.svg?style=plastic" alt="" /></a>-->
[![](https://img.shields.io/badge/Data-10.1101/2022.06.03.494693-green.svg)](https://doi.org/10.1101/2022.06.03.494693)
[![](https://img.shields.io/badge/Preprint-10.1101/2022.06.03.494693-yellow.svg)](https://doi.org/10.1101/2022.06.03.494693)
<!--[![](https://img.shields.io/badge/Data-10.1101/2022.06.03.494693-blue.svg)](https://doi.org/10.1101/2022.06.03.494693)-->
 <!-- badges: end -->

# coupled-oscillators-redox

This repository contains the reproducible code for the generation of data, analysis and figures in the manuscript "Coupling allows robust circadian rhythms despite heterogeneity and noise" ([preprint](https://www.biorxiv.org/)). Note that the simulated data should be generated before (with the *main.py* script or downloaded from [zenodo](https://www.zenodo.org/).

To execute this code:

1. Clone this project at a suitable location. This will place the code within a directory named **coupled-oscillators-redox**
2. Download all the simulated data from ([zenodo](https://www.zenodo.org/)) (or alternatively generate all the simulated data with the *main.py* script)
3. Make sure you have `Python` and basic libraries (`numpy`, `scipy`, `matplotlib`) installed
4. Open the files and start reproducing results

The Python script used to generate the results is *main.py*, which relies on objects from the *poincare.py* script. A solver for stochastic differential equations based on the Euler Maruyama method is also available in the same script as part of the `sdeIntegrator()` class.

To reproduce the figures, the Python files can now be executed in order *fig1.py, fig2.py,* ... within the project.
