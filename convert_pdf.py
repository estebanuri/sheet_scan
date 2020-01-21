import cv2
import numpy as np

# For windows:
# 1. Download zip file with Poppler's latest binaries/dlls from http://blog.alivate.com.au/poppler-windows/ and unzip to a new folder in your program files folder. For example: "C:\Program Files (x86)\Poppler".
# 2. Add "C:\Program Files (x86)\Poppler\poppler-0.68.0\bin" to your SYSTEM PATH environment variable.
from pdf2image import convert_from_path



def pil2cv(pil_image):
    #pil_image = PIL.Image.open('Image.jpg').convert('RGB')
    cv_image = np.array(pil_image)
    # Convert RGB to BGR
    cv_image = cv_image[:, :, ::-1].copy()
    return cv_image


def convert_using_pdf2image():

    #file = "cotton_fields"
    file = "chega_de_saudade"
    full_name = "samples/" + file + ".pdf"
    pages = convert_from_path(full_name, 200, poppler_path='C:/Program Files (x86)/Poopler/poppler-0.68.0/bin')

    # Saving pages in jpeg format
    for i, page in enumerate(pages):

        cv_page = pil2cv(page)
        page_name = "samples/" + file + "_" + str(i) + ".jpg"
        cv2.imwrite(page_name, cv_page)
        cv2.imshow("page", cv_page)
        #page.save('out.jpg', 'JPEG')
        cv2.waitKey()


convert_using_pdf2image()