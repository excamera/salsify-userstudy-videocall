#!/bin/bash

ffmpeg \
    -f rawvideo -video_size 1280x720 -framerate 60 -pixel_format bgra -i beforeFile.raw \
    -f rawvideo -video_size 1280x720 -framerate 60 -pixel_format bgra -i afterFile.raw \
    -filter_complex '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]' \
    -map [vid] \
    -c:v libx264 \
    -crf 23 \
    -preset veryfast \
    concat.mp4
