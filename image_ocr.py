from PIL import Image
import numpy as np
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


def extract_text_from_image(lang_source):
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    im_tmp = Image.open('tmp.jpg')
    text = pytesseract.image_to_string(im_tmp, lang=lang_source)
    im_tmp.close()
    spl_text = text.split("\n")
    text_done = spl_text[0]
    for i in range(len(spl_text) - 1):
        if spl_text[i + 1].isupper() or spl_text[i + 1] == '':
            text_done += f' {spl_text[i + 1]}\n'
        else:
            text_done += f'{spl_text[i + 1]} '

    sentences = text_done.replace('. ', '.\n')
    print(sentences)


def main():
    text_straighten('text-photographed.jpg')
    extract_text_from_image('rus')


if __name__ == "__main__":
    main()
