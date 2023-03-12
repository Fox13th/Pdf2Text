import datetime
import json
import multiprocessing
import pathlib
import re
import easyocr
from PIL import Image
from tqdm import tqdm
from win10toast import ToastNotifier
import numpy as np
import os
import fitz
import cv2
import pytesseract

with open('.\\settings.json', 'r', encoding='utf-8') as f:
    GLOBAL_SETTINGS = json.load()
date_time = datetime.datetime.now()
date_dir = date_time.strftime("%Y-%m-%d")
time_dir = date_time.strftime("%Y-%m-%d-%H-%M-%S")


def get_angle(img):
    img_gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)

    th_box = int(img_gray.shape[0] * 0.007) * 2 + 1
    img_bin_ = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, th_box, th_box)

    img_bin = img_bin_.copy()
    num_rows, num_cols = img_bin.shape[:2]

    best_zero, best_angle = None, 0
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

        if num_zeros > best_zero:
            best_zero = num_zeros
            best_angle = my_angle
    return best_angle * 0.5


def text_straighten(original_img):
    img = cv2.imread(original_img)
    im = Image.open(original_img)
    im_rotate = im.rotate(get_angle(img))
    tmp_save = os.path.join(GLOBAL_SETTINGS['output_dir'], date_dir, f'tmp+{os.getpid()}.jpg')
    im_rotate.save(tmp_save, quality=95)
    im.close()


def extract_text_from_image(output_file="tmp.jpg"):
    output_tmp_file = os.path.join(GLOBAL_SETTINGS['output_dir'], date_dir, f'tmp+{os.getpid()}.jpg')
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    im_tmp = Image.open(output_tmp_file)
    osd = pytesseract.image_to_osd(im_tmp)
    text_image = re.search("Script: ([a-zA-Z]+)\n", osd).group(1)
    if text_image == "Arabic":
        lang_source = 'fas+ara+eng'
    elif text_image == "Cyrillic":
        lang_source = 'rus+eng'
    elif text_image == "Latin":
        lang_source = 'eng'
    else:
        lang_source = f'{text_image.lower()}+eng'

    text = pytesseract.image_to_string(im_tmp, lang=lang_source)
    im_tmp.close()
    os.remove(output_tmp_file)
    spl_text = text.split("\n")
    text_done = spl_text[0]
    for i in range(len(spl_text) - 1):
        if spl_text[i + 1].isupper() or spl_text[i + 1] == '':
            text_done += f' {spl_text[i + 1]}\n'
        else:
            text_done += f'{spl_text[i + 1]} '

    sentences = text_done.replace('. ', '.\n')

    output_file = output_file.replace('.png', '.txt')
    lang_directory = os.path.join(GLOBAL_SETTINGS["output_dir"], date_dir, text_image)
    if not os.path.isdir(lang_directory):
        os.mkdir(lang_directory)

    output_file = os.path.join(lang_directory, os.path.basename(output_file))
    with open(output_file, 'w', encoding='utf-8') as file_for_write:
        file_for_write.write(sentences)
    return lang_directory


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

    page_count = 0

    for page in input_pdf:
        if not page.number == GLOBAL_SETTINGS['pdf_pages'] or GLOBAL_SETTINGS['pdf_pages'] == -1:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            output_file = f"{GLOBAL_SETTINGS['output_dir']}\{date_dir}\{os.path.splitext(os.path.basename(pdf_name))[0]}_page-%i.png" % page.number
            pix.save(output_file)
            output_img.append(output_file)
            page_count = page_count + 1
        else:
            break
    input_pdf.close()

    path_archive_pdf = os.path.join(GLOBAL_SETTINGS['output_dir'], date_dir, 'Archive')
    if not os.path.isdir(path_archive_pdf):
        os.mkdir(path_archive_pdf)

    new_place_pdf = f'{path_archive_pdf}\{os.path.basename(pdf_name)}'
    if os.path.exists(new_place_pdf):
        new_place_pdf = f'{new_place_pdf[:-4]}_{time_dir}.pdf'
    os.rename(pdf_name, new_place_pdf)
    return output_img, page_count


def text_union(file_name_to_union, pages):
    name_united_file = file_name_to_union[:-4]
    if os.path.exists(file_name_to_union):
        file_name_to_union = f'{name_united_file}_{time_dir}.txt'
    with open(file_name_to_union, 'w', encoding='utf-8') as file_finish:
        for page in range(pages):
            part_file = f'{name_united_file}_page-{page}.txt'
            with open(part_file, 'r', encoding='utf-8') as file_page:
                for line in file_page.readlines():
                    file_finish.write(line)
                file_finish.write("\n")
            os.remove(part_file)


def pdf_scans_2_txt(*pdf_files_name):
    for pdf_file in pdf_files_name:
        try:
            if os.path.isfile(pdf_file):
                images, pages_c = convert_pdf_pages_to_img(pdf_file)
                pdf_basename = os.path.basename(pdf_file)
                out_dir_lang = ""
                progress_extraction = tqdm(images)
                for image in progress_extraction:
                    progress_extraction.set_description(f"ФайлЖ {pdf_basename}")
                    text_straighten(image)
                    out_dir_lang = extract_text_from_image(image)
                    os.remove(image)
                text_union(f'{out_dir_lang}\{pdf_basename[:-3]}txt', pages_c)
        except:
            print(f'Возникла ошибка при обработки файла: {pdf_file}')
            path_archive_file = os.path.join(GLOBAL_SETTINGS['output_dir'], date_dir, 'Archive', os.path.basename(pdf_file))
            path_archive_errors = pathlib.Path(os.path.join(GLOBAL_SETTINGS['output_dir'], date_dir, 'Archive', 'Errors'))
            path_archive_errors.mkdir(parents=True, exist_ok=True)
            os.rename(path_archive_file, os.path.join(path_archive_errors, os.path.basename(pdf_file)))


def main():

    pathlib.Path(GLOBAL_SETTINGS['input_dir']).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(GLOBAL_SETTINGS['output_dir'], date_dir)).mkdir(parents=True, exist_ok=True)

    procs = GLOBAL_SETTINGS['num_procs']
    if GLOBAL_SETTINGS['num_procs'] <= 0:
        procs = 1

    new_list = []

    for root, dirs, files in os.walk(GLOBAL_SETTINGS['input_dir']):
        files = [os.path.join(GLOBAL_SETTINGS['input_dir'], files[i]) for i in range(len(files))]
        new_list = [files[i::procs] for i in range(0, len(files)) if not i >= procs]

    list_procs = []
    for i in range(procs):
        p = multiprocessing.Process(target=pdf_scans_2_txt, args=new_list[i])
        list_procs.append(p)
        p.start()

    [proc.join() for proc in list_procs]

    #toaster = ToastNotifier()
    #toaster.show_toast("Распознавание завершено!", f"Распознано документов: {count_docs}", icon_path="images/icon.ico",
    #                   duration=10)


def read_table()
doc = Document('test.docx')
# последовательность всех таблиц документа
all_tables = doc.tables
print('Всего таблиц в документе:', len(all_tables))

# создаем пустой словарь под данные таблиц
data_tables = {i:None for i in range(len(all_tables))}
# проходимся по таблицам
for i, table in enumerate(all_tables):
    print('\nДанные таблицы №', i)
    # создаем список строк для таблицы `i` (пока пустые)
    data_tables[i] = [[] for _ in range(len(table.rows))]
    # проходимся по строкам таблицы `i`
    for j, row in enumerate(table.rows):
        # проходимся по ячейкам таблицы `i` и строки `j`
        for cell in row.cells:
            # добавляем значение ячейки в соответствующий
            # список, созданного словаря под данные таблиц
            data_tables[i][j].append(cell.text)

    # смотрим извлеченные данные 
    # (по строкам) для таблицы `i`
    print(data_tables[i])
    print('\n')

if __name__ == "__main__":
    main()
