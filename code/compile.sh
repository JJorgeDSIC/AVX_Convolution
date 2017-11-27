#!/bin/bash

icc -o sharpen_t sharpen_t.c ctimer.c 
icc -o sharpen_AVX sharpen_AVX.c ctimer.c 

echo "Running both programs..."

./sharpen_t sunset.ppm

mv sharpen.ppm sharpen_baseline.ppm

./sharpen_AVX sunset.ppm

mv sharpen.ppm sharpen_AVX.ppm