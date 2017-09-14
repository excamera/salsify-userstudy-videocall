#!/bin/bash

avconv -f rawvideo -video_size 1280x720 -framerate 60 -pixel_format bgra -i $1 -f yuv4mpegpipe -pix_fmt yuv444p -s 1280x720 -r 60 $2


