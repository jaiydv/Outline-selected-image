import cv2 as cv
from rembg import remove,new_session

from pathlib import Path

def reshape(image):
    """
        this function changes the dimension of original image if it is greater than (1080,720)
        otherwise it will send thw original image

    """
    original_dim = image.shape
    if original_dim[0]>1080 and original_dim[1] > 720:
        dim = (1080, 720)
        image = cv.resize(image, dim, interpolation = cv.INTER_AREA)
    return image

session = new_session()

for file in Path(r'D:\project\outline\TEST IMAGES').glob('*.jpg'): #running over all images in test folder
    input_path = str(file)
    image = cv.imread(input_path)
    image = reshape(image)
    cv.imshow("image",image)
    original_image=image
    while True:
        
        cv.imshow("image",image)
        k=cv.waitKey(0) & 0xFF
        if k == ord('q'):       #Allow the user to do this again and again until "q" is pressed.
            cv.destroyAllWindows()
            break
        if k == ord('c'):       #removes the outline from an image after user presses "c"
            image=original_image
        roi=cv.selectROI(image)

        image_cropped=image[int(roi[1]):int(roi[1]+roi[3]),int(roi[0]):int(roi[0]+roi[2])] #crop the roi image from original image
    
        extracted=remove(image_cropped,session=session)

        grey = cv.cvtColor(extracted, cv.COLOR_BGR2GRAY)
        _, newroi = cv.threshold(grey, 0, 255, cv.THRESH_BINARY)        #converting image into only two colors black and white

        
        cont = cv.findContours(newroi, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) 
        cv.drawContours(image_cropped, cont[0], -1, (0, 255, 0),thickness=3)  #drawing contours 
        cv.destroyAllWindows()
        