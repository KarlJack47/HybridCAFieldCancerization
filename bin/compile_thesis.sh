#!/bin/bash

cd MSC_Thesis

pdflatex thesis.tex

rm *.aux\
   *.bcf\
   *.log\
   *.out\
   *.run.xml\
   *.toc

exit 0
