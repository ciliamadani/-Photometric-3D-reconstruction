import numpy as np 
import cv2 

def load_images():

    return

def load_objMask():
    path =  dataset_dir+"/mask.png"
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    return img


def load_intensSources():

    path =  "C:/Users/AzurComputer/Desktop/M2/computer vision/tp/projet/objet1PNG_SII_VISION/light_intensities.txt"
    intensity_matrix = np.ndarray((96,3))
    all = []
    with open(path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            row = list(line.strip().split(" "))
            
            float_row = []
            for item in row:
                float_row.append(float(item))
        
            all.append(float_row)
            
    intensity_matrix = np.array(all)
    return intensity_matrix

def load_images():
    #n nombre d'image a importer (max 96)
    i=0
    n=3 ## Just to test ... 
    intensSourcesMatrix = load_intensSources()
    file= open("C:/Users/AzurComputer/Desktop/M2/computer vision/tp/projet/objet1PNG_SII_VISION/filenames.txt",'r')  
    val = pow(2,16)-1
    #depending on data
    h=512
    w=612
    imagesMatrix=np.ndarray((96,h*w*3))
    for i in range(n):
        #read the file that contains the images name (images in 16bits)
        #read one image -> get the name from filenames.txt
        imageName= file.readline()
        print("treating image : " + imageName)
        pathToImage= str('C:/Users/AzurComputer/Desktop/M2/computer vision/tp/projet/objet1PNG_SII_VISION/'+imageName).strip()
        image = cv.imread(pathToImage,-1)
        if image is None: 
            print("image is none")
        else:
            #changer d'intrevale 
            imageRes=rescale(image)

            ## out of loop 
            m = load_intensSources()
            ## 

            print(m[i])
            intensite  = [m[i][0] ,m[i][1],m[i][2]]
            imageRes=divIntensite(imageRes,intensite)
            imageRes=imageToOneLine(imageRes)

            # check shape 
            imagesMatrix[i]=imageRes
            #print('here')
            #print(imageRes.shape)
    print(imagesMatrix.shape)
    return imagesMatrix
            

def load_lightSources():

    path =  dataset_dir+"/light_intensities.txt"
    intensity_matrix = np.ndarray((96,3))
    
    with open(path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            row = list(line.strip().split(" "))

            int_row = []
            for item in row:
                float_row.append(float(item)) 

            row = np.array(float_row)
            intensity_matrix = np.append(intensity_matrix, [row], axis=0)
        
    return intensity_matrix

def rescale(image):
    return image.astype('float32') / (2**16 -1)

def divIntensite(image , intensite):
    # l’intensité (B/intB, G/intG, R/intR)
    #intenisté array of 3 values ( int B , int G , int R)
    #image X , Line X dans fichier d'intensité 
    h = image.shape[0]
    w = image.shape[1]
    for y in range(h):
        for x in range(w):
        
            image[y,x][0]=image[y,x][0]/intensite[0]
            image[y,x][1]=image[y,x][1]/intensite[1]
            image[y,x][2]=image[y,x][2]/intensite[2]
            
    return image
    

def image_to_greyScale(image):
    #replace each pixel ( array ) with 1 value using this formula (NVG = 0.3 * R + 0.59 * G + 0.11 * B)
    #pixel in opencv is bgr
    h=image.shape[0]
    w=image.shape[1]
    newImage=np.zeros((h,w),np.float32)
    for y in range(h):
        for x in range(w):
            newImage[y,x]=image[y,x][0]*0.11 + image[y,x][1]*0.59 + image[y,x][2]*0.32
    return newImage


def imageToOneLine(image):
    newImage = image.reshape(1,-1)
    return newImage


"""
PART 2 
"""

def calcul_needle_map():
    obj_images = load_images()
    light_sources = load_lightSources()
    obj_masques = load_objMask() # ???

    # calcul de la matrice light_sources pseudo inverse 
    pinv_light_sources = numpy.linalg.pinv(light_sources)

    # faire le produit matriciel pour trouver les vecteurs normals 
    N = numpy.dot(pinv_light_sources,obj_images)

    return N
if __name__ == '__main__':
    print('Please give path to the project director: /n')
    dataset_dir = input()
    matrix = load_lightSources()
    print(matrix)