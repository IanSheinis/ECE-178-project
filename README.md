﻿# ECE-178-project
Was given 500 images of a blurred, rotated, and translated molecule in .tif fomat and was tasked with removing the blur to get a clear image.
I first used a tophat filter to remove some blur, then I translationally aligned the molecules using fft convolution, 
then I rotationally aligned the images using fft covolution in polar coordinates and then image averaged all the files.
