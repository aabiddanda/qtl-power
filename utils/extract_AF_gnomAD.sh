#!/bin/bash

vcf_file=$1
pop=$2 # (should be afr, nfe, seu, sas for example)

# Create a temporary header
echo -e "ID\tCHROM\tPOS\tREF\tALT\tAC_${pop}\tAN_${pop}\tAF_${pop}\tvep" > /tmp/tmp.header.txt

# Create a file with the appropriate output
bcftools query -f "%ID\t%CHROM\t%POS\t%REF\t%ALT\t%AC_${pop}\t%AN_${pop}\t%AF_${pop}\t%vep\n" ${vcf_file} | awk '$6 > 0' | cat /tmp/tmp.header.txt -
