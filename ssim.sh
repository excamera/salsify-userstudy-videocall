#!/bin/bash

for dir in $(ls -d */); do
    cd $dir
    echo $dir
    if [[ ! -f ssim.log ]]; then
	    dump_ssim -p 8 before1.y4m after1.y4m | tee ssim1.log
	    dump_ssim -p 8 before2.y4m after2.y4m | tee ssim2.log
    fi
    cd ..
done
