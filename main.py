import numpy as np 
import cv2 as cv  
import math
import matplotlib.pyplot as plt 

def load_intensSources():

    path =  proj_dir+"/light_intensities.txt"
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
    print("Intensities loaded ... ")
    return intensity_matrix

def load_objMask():
    path =  proj_dir+"/mask.png"
    img = cv.imread(path, cv.IMREAD_UNCHANGED)

    return img

def load_images():
    #n nombre d'image a importer (max 96)
    i=0
    n=96 ## Just to test ... 
    intensSourcesMatrix = load_intensSources()
    file= open(proj_dir+"/filenames.txt",'r')  
    val = pow(2,16)-1
    #depending on data
    h=512
    w=612
    imagesMatrix=np.ndarray((96,h*w))
    m = load_intensSources()
    
    for i in range(n):
        #read the file that contains the images name (images in 16bits)
        #read one image -> get the name from filenames.txt
        imageName= file.readline()
        #print("treating image : " + imageName)
        pathToImage= str(proj_dir+imageName).strip()
        image = cv.imread(pathToImage,-1)
        if image is None: 
            print("image is none")
        else:
            #changer d'intrevale 
            imageRes=rescale(image)
            intensite  = [m[i][0] ,m[i][1],m[i][2]]
            imageRes=divIntensite(imageRes,intensite)
            imageRes = image_to_greyScale(imageRes)
            imageRes= imageToOneLine(imageRes)
            imagesMatrix[i]=imageRes
            
    print("Images loaded ... ")
    return imagesMatrix
            
            
    
#positions des sources lumineuses
def load_lightSources():
    path=(proj_dir+"/light_directions.txt")
    i=0
    all = []
    
    with open(path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            row = list(line.strip().split(" "))
            float_row = []

            for item in row:
                float_row.append(float(item))
            all.append(float_row) 


    light_directionsMatrix = np.array(all)

    print("Light sources loaded ... ")
    return light_directionsMatrix

def rescale(image):
    return image.astype('float32') / (2**16 -1)

def divIntensite(image , intensite):
    # l’intensité (B/intB, G/intG, R/intR)
    #intenisté array of 3 values ( int B , int G , int R)
    #image X , Line X dans fichier d'intensité 

    for c in range(3):
        image[:,:, 2-c] = image[:,:,c] /intensite[c]
    
    return image
    
def image_to_greyScale(image):
    #replace each pixel ( array ) with 1 value using this formula (NVG = 0.3 * R + 0.59 * G + 0.11 * B)
    #pixel in opencv is bgr
    h=image.shape[0]
    w=image.shape[1]
    newImage=np.zeros((h,w),np.float32)
    
    lfunc = lambda x:  x[:, :,0]*0.11 + x[:, :,1]*0.59 + x[:, :,2 ]*0.32
    newImage  = lfunc(image)
    return newImage

def imageToOneLine(image):
    newImage = image.reshape(1,-1)
    return newImage


def calcul_needle_map():
    obj_images = load_images() # load a matrix, each line has an image, each col a pixel
    light_sources = load_lightSources()  # load matrix of light sources, each line takes a light direction x y z
    obj_masques = load_objMask() # load an img matrix

    # calcul de la matrice light_sources pseudo inverse 
    pinv_light_sources = np.linalg.pinv(light_sources)

    # faire le produit matriciel pour trouver les vecteurs normals 
    N = np.dot(pinv_light_sources,obj_images)


    # create the final matrix 
    h=512
    w=612
    k = 0
    finalM  = np.ndarray((h,w,3))

    for i in range(h):
        for j in range(w):
            finalM[i,j] = N[:,k]
            k = k+1

    print("needle map calculated ... ")
    return  finalM

def normalize(img):
    for i in range (512):
        for j in range (612):
            nx = img[i,j][0]
            ny = img[i,j][1]
            nz = img[i,j][2]
            N_len =  math.sqrt(nx*nx + ny*ny +nz*nz)
            img[i,j] = [nx/N_len, ny/N_len, nz/N_len] 

    print("Normalisation done ...")

    return img 

def changeInterval(img):
    for i in range (512):
        for j in range (612):
            for k in range (3):
                x = img[i,j][k]
                img[i,j][k] = ((x+1)/2)*255

    print("Changed interval ... ")

    return img 



if __name__ == '__main__':
    print('Please give path to the project director: /n')
    proj_dir = input()

    normalVects = calcul_needle_map()
    normalVects = normalize(normalVects)
    normalVects = changeInterval(normalVects)
    int_array = normalVects.astype(int)

    cv.imwrite(proj_dir+'imgNormal.png', int_array)
    img = cv.imread(proj_dir+'imgNormal.png')

    cv.imshow('normals',img)
    cv.waitKey(0)
    cv.destroyAllWindows()


    ## apply mask 
    obj_masques = load_objMask() # load an img matrix
    for i in range (512):
        for j in range (612):
            # if pix is black 
            if  obj_masques[i,j] == 0:
                # normal vect blacked 
                int_array[i,j] = [0,0,0]

    cv.imwrite(proj_dir+'imgNormalMask.png', int_array)
    img = cv.imread(proj_dir+'imgNormalMask.png')
    cv.imshow('normals after mask',img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    print("Finished.")




    
    
