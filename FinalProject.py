import PIL.Image
import numpy as np
from skimage import io
from scipy.ndimage import median_filter
from scipy.ndimage import gaussian_filter
from skimage.restoration import denoise_wavelet
from skimage.morphology import white_tophat, disk, black_tophat
import argparse
import matplotlib.pyplot as plt
import PIL 
import numpy as np
from skimage import io
import matplotlib.pyplot as plts
import skimage
from PIL import Image as im 
import polarTransform as pt
import cv2 as cv
def tophat(image_path, output_path, radius = 15):
    
    image = io.imread(image_path)

    if image.ndim == 3:
        image = np.mean(image, axis=-1).astype(np.float32)
    structuring_element = disk(radius)

    tophat_filtered_image = white_tophat(image, structuring_element)
    tophat_filtered_image = (tophat_filtered_image - tophat_filtered_image.min()) /(tophat_filtered_image.max() - tophat_filtered_image.min())

    if output_path:
        io.imsave(output_path, (tophat_filtered_image * 255).astype(np.uint8))


def shift(imgShift, image1, image2, output_path = None):
    shift = tuple(map(int, imgShift))
    outputimg = np.roll(image2, shift = shift, axis = (0,1))
    outputimg = im.fromarray(outputimg.astype(np.uint8))
    # if output_path:
    #     outputimg.save(output_path)
    image1 = np.array(image1)
    image2 = np.array(image2)

    image1 = cv.cvtColor(image1, cv.COLOR_BGR2RGB)
    image2 = cv.cvtColor(image2, cv.COLOR_BGR2RGB)
    
    outputimg = np.array(outputimg)
    translatedImage = cv.cvtColor(outputimg, cv.COLOR_BGR2RGB)

    outputimg = PIL.Image.fromarray(outputimg)
    if output_path:
        outputimg.save(output_path)

    # Plot the images side by side
    plt.figure(figsize=(15, 5))  # Set figure size (width, height)

    # Plot image1
    plt.subplot(1, 3, 1)  # 1 row, 3 columns, first position
    plt.imshow(image1)
    plt.title('Image 1')
    plt.axis('off')  # Turn off axis

    # Plot image2
    plt.subplot(1, 3, 2)  # 1 row, 3 columns, second position
    plt.imshow(image2)
    plt.title('Image 2')
    plt.axis('off')  # Turn off axis

    # Plot translatedImage
    plt.subplot(1, 3, 3)  # 1 row, 3 columns, third position
    plt.imshow(translatedImage)
    plt.title('Translated Image')
    plt.axis('off')  # Turn off axis

    # Show the plot
    plt.tight_layout()  # Adjust spacing
    plt.show()

def radMask(index,radius,array):
    a,b = index
    nx,ny = array.shape
    y,x = np.ogrid[-a:nx-a,-b:ny-b]
    mask = x*x + y*y <= radius*radius

    return mask

def fastconv(img1_path,img2_path, output_path = None):
    image1 = cv.imread(img1_path, cv.IMREAD_GRAYSCALE)
    image2 = cv.imread(img2_path, cv.IMREAD_GRAYSCALE)

    img1 = np.array(image1)
    img2 = np.array(image2)

    img1 = np.fft.fft2(img1)
    img2 = np.fft.fft2(img2)

    outputArray = img1*img2

    
    
    outputArray = np.fft.ifft2(outputArray)
    outputArray = np.abs(outputArray)
    outputArray = np.fft.ifftshift(outputArray)
    indices = np.where(outputArray == outputArray.max())
    indices_y = indices[0][0]
    indices_x = indices[1][0]

    indices_y -= 400
    indices_x -= 400
    #print(indices_y)
    #print(indices_x)

    shift((-indices_y,-indices_x), image1, image2, output_path)
    #plotImage3(image1, image2, outputArray)

def plotImage(arr):
    # Plot the images side by side
    plt.figure(figsize=(15, 5))  # Set figure size (width, height)

    # Plot image1
    #plt.subplot(1, 3, 1)  # 1 row, 3 columns, first position
    plt.imshow(arr)
    plt.title('Image 1')
    plt.axis('off')  # Turn off axis

    plt.show()

def plotImage3(image1,image2,translatedImage):
    # Plot the images side by side
    plt.figure(figsize=(15, 5))  # Set figure size (width, height)

    # Plot image1
    plt.subplot(1, 3, 1)  # 1 row, 3 columns, first position
    plt.imshow(image2)
    plt.title('Image 2')
    plt.axis('off')  # Turn off axis

    # Plot image2
    plt.subplot(1, 3, 2)  # 1 row, 3 columns, second position
    plt.imshow(image1)
    plt.title('Image 1')
    plt.axis('off')  # Turn off axis

    # Plot translatedImage
    plt.subplot(1, 3, 3)  # 1 row, 3 columns, third position
    plt.imshow(translatedImage)
    plt.title('Translated Image')
    plt.axis('off')  # Turn off axis

    # Show the plot
    plt.tight_layout()  # Adjust spacing
    plt.show()

def average_images(image_path1, image_path2, output_path = None):
    img1 = cv.imread(image_path1)
    img2 = cv.imread(image_path2)

    averaged_img = cv.addWeighted(img1, 0.85, img2, 0.15, 0)

    plotImage3(img1,img2,averaged_img)
    if output_path:
        cv.imwrite(output_path, averaged_img)

def orientationalShift(img1_path, img2_path, output_path = None):
    image1 = cv.imread(img1_path, cv.IMREAD_GRAYSCALE)
    image2 = cv.imread(img2_path, cv.IMREAD_GRAYSCALE)

    pimage1, ptSettings1 = pt.convertToPolarImage(image1, center = [400,400], hasColor = True)
    pimage2, ptSettings2 = pt.convertToPolarImage(image2, center = [400,400], hasColor = True)

    img1 = np.array(pimage1.T)
    img2 = np.array(pimage2.T)

    img1 = np.fft.fft2(img1)
    img2 = np.fft.fft2(img2)

    outputArray = img1* np.conjugate(img2)

    
    
    outputArray = np.fft.ifft2(outputArray)
    outputArray = np.abs(outputArray)
    outputArray = np.fft.ifftshift(outputArray)

    indices = np.where(outputArray == outputArray.max())

    indices_y = indices[0][0] 
    #indices_y = 800
    indices_x = indices[1][0]


    print(indices_y)
    print(indices_x)
    imgShift = (-indices_y,-indices_x)
    shift = tuple(map(int, imgShift))
    outputimg = np.roll(pimage2, shift = shift, axis = (0,1))
    
    outputimg = ptSettings2.convertToCartesianImage(outputimg)
    #plotImage3(image1,image2,outputimg)
    plotImage(outputArray)

    outputimg = PIL.Image.fromarray(outputimg)
    if output_path:
        outputimg.save(output_path)

if __name__ == "__main__":
    # for i in range(1,100):
    #     print(i)
    #     zeros = '00'
    #     if(i >=10): zeros = '0'
    #     if(i>=100): zeros = ''
    #     tophat("img" + zeros + str(i) + ".tif", "atophat" + str(i) + ".tif",15)  
    # tophat("ablackhat7.tif", "ablacktophat7.tif")  
    # normalize_image("img007.tif", "normalize007.tif")


    # for i in range(1,100):
    #     print(i)
    #     zeros = '00'
    #     if(i >=10): zeros = '0'
    #     if(i>=100): zeros = ''
    #     crosscorrelate("atophat4.tif", "atophat" + str(i) + ".tif", "atrans" + str(i) + ".tif")

    #phaseCorrelate("img004.tif","img007.tif","aaa.tif")
    # for i in range(1,100):
    #     print(i)
    #     fastconv("atophat4.tif","atophat"+str(i)+".tif","atrans" + str(i) + ".tif")

    #fastconv("atophat4.tif", "aor43.tif","afinish43.tif")
    orientationalShift("atrans4.tif","atrans90.tif","aor7.tif")
    #average_images("acomplete34.tif","afinish43.tif","acomplete43.tif")

