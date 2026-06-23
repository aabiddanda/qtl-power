
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/aabiddanda/qtl-power/HEAD) [![Read the Docs](https://img.shields.io/readthedocs/qtl-power)](https://aabiddanda.github.io/qtl-power/)

# Power Calculation Routines for GWAS Study Design

This is a library to quickly calculate power curves for the expected detected effect in Genome-Wide Association Studies.

## Installation

If you are interested in installing this package for direct use within scripts or notebooks, please run:

```
pip install qtl-power
```

## Interactive Exploration via Notebooks

If you are primarily interested in a more interactive experience, you can immediately use several of our pre-built notebooks via the [`mybinder`](https://mybinder.org/v2/gl/data-analysis5%2Fqtl-power/default?labpath=notebooks%2F) link above. This will allow you to use the library to generate commonly used plots for comparing power for genetic association based on mutliple input parameters.

## Documentation

Currently the documentaton is held in the `/docs` directory and is built using [`Sphinx`](https://www.sphinx-doc.org/en/master/). To rebuild the documentation:

```
cd docsrc
make clean html copy
cd ..
git add docs/
```

then create a commit that will create an updated set of documentation.

## References

* Matti Pirinen [(GWAS Course Notes)](https://www.mv.helsinki.fi/home/mjxpirin/GWAS_course/material/GWAS3.html)
* Andriy Derkach, Haoyu Zhang, Nilanjan Chatterjee, Power Analysis for Genetic Association Test (PAGEANT) provides insights to challenges for rare variant association studies, Bioinformatics, Volume 34, Issue 9, 01 May 2018, Pages 1506–1513, [https://doi.org/10.1093/bioinformatics/btx770](https://doi.org/10.1093/bioinformatics/btx770)
* Jennifer Li Johnson, Goncalo Abecasis [GAS Calculator](https://github.com/jenlij/GAS-power-calculator/blob/master/equations_gas_power_calc.pdf)
