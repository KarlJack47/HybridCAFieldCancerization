#!/bin/bash

cd MSc_Thesis

pdflatex thesis.tex > /dev/null
biber thesis > /dev/null
pdflatex thesis.tex > /dev/null
pdflatex thesis.tex > /dev/null

rm *.aux *.bcf *.log *.out *.run.xml *.toc *.bbl *.blg *.lot *.lof

echo 'The thesis was successfully compiled.'

exit 0