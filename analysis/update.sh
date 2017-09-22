#!/bin/bash

./delay.py
./quality.py

./svg2pdf

cp delay.pdf ../../salsify-paper/figures/userstudy-videocall-delay.pdf
cp quality.pdf ../../salsify-paper/figures/userstudy-videocall-quality.pdf 

cd ../../salsify-paper/figures

rm userstudy-videocall-delay-crop.pdf
rm userstudy-videocall-delay-quality.pdf

pdfcrop userstudy-videocall-delay.pdf
pdfcrop userstudy-videocall-quality.pdf

make -C ..
