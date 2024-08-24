# Computer Vision #

This repository contains two Python scripts designed to process and analyze PNG images for different purposes.

## Files ##

1. **`TP_1.py`**
   - This script processes a single PNG file containing hidden objects. By applying various image transformations such as thresholding, grayscale conversion, contrast adjustment, and brightness enhancement, the script identifies and highlights the hidden objects within the image.

2. **`TP_2.py`**
   - This script is designed to handle multiple PNG files that represent scanned multiple-choice exams. It automates the validation of student data (including name, code, ID, and date) and facilitates the automated grading of the exams. The script reads the scanned exam images, extracts relevant information, and processes the answers to provide quick and accurate results.


## **Execution:** ##

To use `pdf2image` (necessary for TP_2) in Visual Studio Code, you need to install **Anaconda**.
Set it as the interpreter in Visual Studio Code.
Then, run the following command:
```bash
conda install -c conda-forge poppler
```

Additionally, to use OCR in Visual Studio Code, follow these steps:

1. First, install the Tesseract installer for Windows. You can find it at the following link: [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki).


2. Run the following commands in the terminal. Add "py -m -v.v" at first if you have more than one python version installed in your computer.

```bash
pip install Pillow pytesseract
```

Certain libraries need to be imported for functionality, but these are already included in the code.
There is also a ipynb notebook provided where the OCR item is executed, in case you prefer not to install OCR on your computer.