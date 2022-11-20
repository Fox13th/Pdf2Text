from setuptools import setup, find_packages

setup(name='Pdf2Txt',
      version='0.21',
      url='https://github.com/Fox13th/Pdf2Text',
      license='fox.inc',
      author='FoxyAn',
      author_email='narnianlion@yandex.ru',
      description='Recognition PDF files and show them as txt',
      packages=find_packages(exclude=['PIL', 'numpy', 'fitz', 'cv2', 'pytesseract']),
      zip_safe=False)