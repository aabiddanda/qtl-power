# Power Calculation Routines for GWAS Study Design

## GWAS Power Calculations

In the context of GWAS and detection power, we are interested in a number of questions that each can be encapsulated by a specific plot:

1. What is the sample-size required to detect a causal effect of this size at a given minor allele frequency (MAF)?
2. What effect-size is detectable at a specific power level (e.g. 80% power) with a given sample size?

These questions can subsequently be used for asking more layered questions for study design, such as at what MAF regimes will one study be more strongly powered over another.

### Key Parameters

* Effect-Size
* Sample Size (N)
* MAF (minor allele frequency)
* Imputation r2 (default r2 > 0.95)

For Case/Control traits, the proportion of cases is also required for the power claculations.

## Rare-Variant Association Power Calculations

Rare variant association tests are often designed to overcome the lack of power for association at individual variants by aggregating variants across a region (e.g. a gene). The two primary tests are "Burden" tests, which are well-powered well all of the effects in a gene are in the same direction on a trait, and "Variance Component" tests, which are powerful when the rare effects drive larger variance in the trait.

In order to appropriately account for variation in the number of exonic variation across genes, and the frequency of those variants we resample both of these quantities from distributions fit to the MLE of a normal distribution and a gamma distribution repectively on Chromosome 4 of the GnomeAD data Release. These are the default parameters used throughout the notebooks and represent a reasonable approximation.

All of the results

### Key Parameters

* J: number of variants
* MAF: vector of minor allele frequency
* TEV: total explained variance


## Cis / Trans QTL Power Calculations & Differential Expression Power

**TODO: need to specify this.**

## References

* Matti Pirinen [(GWAS Course Notes)](https://www.mv.helsinki.fi/home/mjxpirin/GWAS_course/material/GWAS3.html)
* Andriy Derkach, Haoyu Zhang, Nilanjan Chatterjee, Power Analysis for Genetic Association Test (PAGEANT) provides insights to challenges for rare variant association studies, Bioinformatics, Volume 34, Issue 9, 01 May 2018, Pages 1506–1513, [https://doi.org/10.1093/bioinformatics/btx770](https://doi.org/10.1093/bioinformatics/btx770)
* Jennifer Li Johnson, Goncalo Abecasis [GAS Calculator](https://github.com/jenlij/GAS-power-calculator/blob/master/equations_gas_power_calc.pdf)
