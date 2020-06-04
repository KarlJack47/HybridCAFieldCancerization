#!/bin/bash

cd MSC_Thesis

pdflatex thesis.tex > /dev/null
biber thesis > /dev/null
pdflatex thesis.tex > /dev/null
pdflatex thesis.tex > /dev/null

rm *.aux *.bcf *.log *.out *.run.xml *.toc *.bbl *.blg

exit 0