import cv2
import numpy as np
from pdf2image import convert_from_path

def pil2cv(pil_image):
    #pil_image = PIL.Image.open('Image.jpg').convert('RGB')
    cv_image = np.array(pil_image)
    # Convert RGB to BGR
    cv_image = cv_image[:, :, ::-1].copy()
    return cv_image

file = "cotton_fields"
full_name = "samples/" + file + ".pdf"
pages = convert_from_path(full_name, 200)

# Saving pages in jpeg format
for i, page in enumerate(pages):

    cv_page = pil2cv(page)
    page_name = "samples/" + file + "_" + str(i) + ".jpg"
    cv2.imwrite(page_name, cv_page)
    cv2.imshow("page", cv_page)
    #page.save('out.jpg', 'JPEG')
    cv2.waitKey()
