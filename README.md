<!-- badges: start -->
<a href="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205155"><img src="https://img.shields.io/badge/Data-GSE205155-green.svg?style=plastic" alt="" /></a>
[![](https://img.shields.io/badge/bioRxiv-10.1101/2022.06.03.494693-yellow.svg)](https://doi.org/10.1101/2022.06.03.494693)
[![](https://img.shields.io/badge/Data-10.1101/2022.06.03.494693-blue.svg)](https://doi.org/10.1101/2022.06.03.494693)
[![DOI](https://zenodo.org/badge/541140885.svg)](https://zenodo.org/badge/latestdoi/541140885)
 <!-- badges: end -->

# coupled-oscillators-redox

This repository contains the reproducible code for the generation of data, analysis and figures in the manuscript "Coupling allows robust circadian rhythms despite heterogeneity and noise" ([preprint](https://www.biorxiv.org/)). 

To execute this code:

1. Clone this project at a suitable location. This will place the code within a    directory named **coupled-oscillators-redox**
2. Make sure you have `Python` and basic libraries (`numpy`, `scipy`, `matplotlib`) installed
3. Open the files and start reproducing results

The Python script used to generate the results is *main.py*, which relies on objects from the *poincare.py* script.

To reproduce the figures, the Python files can now be executed in order *fig1.py, fig2.py,* ... within the project.
