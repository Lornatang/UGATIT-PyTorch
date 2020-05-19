#!/bin/bash

wget https://drive.google.com/uc?export=download&confirm=pO_n&id=1xOWj1UVgp6NKMT3HbPhBbtq2A4EDkghF
mkdir selfie2anime
mkdir selfie2anime/train
mkdir selfie2anime/test
mkdir selfie2anime/train/A
mkdir selfie2anime/train/B
mkdir selfie2anime/test/A
mkdir selfie2anime/test/B

unzip selfie2anime.zip

mv trainA/* selfie2anime/train/A
mv trainB/* selfie2anime/train/B
mv testA/* selfie2anime/test/A
mv testB/* selfie2anime/test/A

rm -rf trainA
rm -rf trainB
rm -rf testA
rm -rf testB

rm selfie2anime.zip
