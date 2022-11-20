import easyocr
from PIL import Image
import numpy as np
import os
import fitz
import cv2
import pytesseract


def get_angle(img):
    # сперва переведём изображение из RGB в чёрно серый
    # значения пикселей будут от 0 до 255
    img_gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)

    # а теперь из серых тонов, сделаем изображение бинарным
    th_box = int(img_gray.shape[0] * 0.007) * 2 + 1
    img_bin_ = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, th_box, th_box)

    img_bin = img_bin_.copy()
    num_rows, num_cols = img_bin.shape[:2]

    best_zero, best_angle = None, 0
    # итеративно поворачиваем изображение на пол градуса
    for my_angle in range(-20, 21, 1):
        rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), my_angle / 2, 1)
        img_rotation = cv2.warpAffine(img_bin, rotation_matrix, (num_cols * 2, num_rows * 2),
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=255)

        img_01 = np.where(img_rotation > 127, 0, 1)
        sum_y = np.sum(img_01, axis=1)
        th_ = int(img_bin_.shape[0] * 0.005)
        sum_y = np.where(sum_y < th_, 0, sum_y)

        num_zeros = sum_y.shape[0] - np.count_nonzero(sum_y)

        if best_zero is None:
            best_zero = num_zeros
            best_angle = my_angle

        # лучший поворот запоминаем
        if num_zeros > best_zero:
            best_zero = num_zeros
            best_angle = my_angle
    return best_angle * 0.5


def text_straighten(original_img):
    img = cv2.imread(original_img)
    im = Image.open(original_img)
    im_rotate = im.rotate(get_angle(img))
    im_rotate.save('tmp.jpg', quality=95)
    im.close()


def extract_text_from_image(lang_source, output_file="tmp.jpg"):
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    im_tmp = Image.open('tmp.jpg')
    text = pytesseract.image_to_string(im_tmp, lang=lang_source)
    im_tmp.close()
    os.remove('tmp.jpg')
    spl_text = text.split("\n")
    text_done = spl_text[0]
    for i in range(len(spl_text) - 1):
        if spl_text[i + 1].isupper() or spl_text[i + 1] == '':
            text_done += f' {spl_text[i + 1]}\n'
        else:
            text_done += f'{spl_text[i + 1]} '

    sentences = text_done.replace('. ', '.\n')

    if not output_file == "tmp.jpg":
        output_file = output_file.replace(".png", ".txt")
    else:
        output_file = output_file.replace(".jpg", ".txt")

    with open(output_file, 'w', encoding='utf-8') as file_for_write:
        file_for_write.write(sentences)


def recognition_text_easyocr(lang_source, output_file="tmp.jpg"):
    reader = easyocr.Reader(["ru", "en"])
    result = reader.readtext('tmp.jpg', detail=0, paragraph=True)
    if not output_file == "tmp.jpg":
        output_file = output_file.replace(".png", ".txt")
    else:
        output_file = output_file.replace(".jpg", ".txt")

    with open(output_file, "w", encoding='utf-8') as file:
        for line in result:
            file.write(f"{line}\n\n")

    return f"Result wrote into {output_file}"


def convert_pdf_pages_to_img(pdf_name, how_many_pages, output_dir):
    zoom_x = 2
    zoom_y = 2
    mat = fitz.Matrix(zoom_x, zoom_y)

    input_pdf = fitz.open(pdf_name)
    output_img = []
    for page in input_pdf:
        if not page.number == how_many_pages:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            output_file = f"{output_dir}\{os.path.splitext(os.path.basename(pdf_name))[0]}_page-%i.png" % page.number
            pix.save(output_file)
            output_img.append(output_file)
        else:
            break
    input_pdf.close()
    return output_img


def pdf_scans_2_txt():
    images = convert_pdf_pages_to_img("table.pdf", 4, r'C:\Users\deanw\PycharmProjects\sdada')
    for image in images:
        text_straighten(image)
        extract_text_from_image('rus+eng', image)
        #recognition_text_easyocr("en", image)
        os.remove(image)


def main():
    #text_straighten('img_text.jpg')
    #extract_text_from_image('rus')
    #print(recognition_text_easyocr("en", "text-photographed.jpg"))
    pdf_scans_2_txt()

if __name__ == "__main__":
    main()
