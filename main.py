import numpy as np 
import cv2 

def load_images():

    return

def load_objMask():
    path =  dataset_dir+"/mask.png"
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    cv2.imshow('mask',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img


def load_intensSources():

    return 

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

def image_to_greyScale():


    return


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