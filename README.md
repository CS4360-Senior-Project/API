# Capstone Project- Kataba
(( Handwriting Recognition and Handwriting Sample Comparison ))

## Instructions to run local server:

Install required python packages:
```
cd API
pip install -r requirements.txt
```

To run the server:
```
cd config
python manage.py runserver
```

Once you compile and run the server, open http://127.0.0.1:8000/.
You can navigate to the main page at: http://127.0.0.1:8000/main-page/

Every time you make changes to models (database tables) you need to make migrations to the database
```
python manage.py makemigrations
python manage.py migrate
```


## Google Colab Notebook Links: 

### Jada Sachetti- 
The following notebooks contain code that is deployed on our web app for the handwriting analysis:

Image Cropping https://colab.research.google.com/drive/19pK4YeHhuI_DLjxxy74lYXk4KzhUBr9X?usp=sharing

Feature Extraction and Model implementation https://colab.research.google.com/drive/1FYU0FcZU6zAjOhbmENlXJb4Oyi0Mp3J6#scrollTo=f5PhhAtigNbU

Image Comparison (post Siamese model training) https://colab.research.google.com/drive/1trMhaqYBv0JYmTX6iFH-xqS3goO4MYb0?usp=sharing

Pytesseract addition page code https://colab.research.google.com/drive/1Btr0bTCvM0QXPfe7ez1S3HOLG9iCVNuk#scrollTo=_K6MzRl26-Vs

These are other notebooks that were used for trial:

OCR and Pytesseract https://colab.research.google.com/drive/1xzRLc0nFE3g5CQJtGGQHEfzDwFQ1Gvej#scrollTo=Gu1ICF_SwzBY

More OCR Attempts with pretrained models https://colab.research.google.com/drive/1MsNAzIxSttAJEVZ9fkD4ohU9WC_TeMWR

"Last Hope" notebook includes new trials of image preprocessing and Siamese model implementation https://colab.research.google.com/drive/1rO6S7h_EgXcC0E5VAvCQ69TFSHC7XDlj

### Emily Carrillo- 

Dataset Organization (Creating files for each writer and each with a validation set) https://colab.research.google.com/drive/1XItAN5PuMnOofUvfJoUlayD-eJDsRYgL?authuser=1#scrollTo=58pLWsAqKqWY

Rough Drafts

Dataset Organization Rough Drafts (Using CSV and labels) Part 1 , Part 2 , and Part 3  https://colab.research.google.com/drive/1-BCoZ3U43Ahq4fAbDvw5t4TuteGk89hX?authuser=1  /  https://colab.research.google.com/drive/1Y6ynJ-m1eEEjbEGy8HwR1NpbKgAnVA75?authuser=1 /  https://colab.research.google.com/drive/1AgLx5E_RiRp17UGFsZcmAUaXQIBtsQDe?authuser=1

Character Segmentation Rough Drafts / Test Part 1 and Part 2 https://colab.research.google.com/drive/1Anld83J6-UGn7evuv_NCBACURNdh1AQO?authuser=1 /  https://colab.research.google.com/drive/1hhS9ACI74dELDGFa5LOkbYAdGOyKPXU5?authuser=1

Image Processing Rough Drafts / Test Part 1 , Part 2 , and Part 3 https://colab.research.google.com/drive/1tyh3cg-UqW4n386KUpFpx0Mb6fyhB3O-?authuser=1  /  https://colab.research.google.com/drive/1qLup11nLxwqKxYYjvpYhI0Ci6GMdBqL6?authuser=1#scrollTo=zNil6_xp3xtf / https://colab.research.google.com/drive/1xW1o7ZPrZt1PwGjvCXRl_OUUwiU2YVR_?authuser=1

Model Training Using 3 column csv ( positive, negative, anchor ) Part 1 and Part 2 https://colab.research.google.com/drive/1a8khyZHmwRNHLLn-2OdU7QHHaMberpPl?authuser=1#scrollTo=Dff2SeP75WCO  /  https://colab.research.google.com/drive/1q2ANNNi5j-GcOZ3OEbVqIbjw6851BjOl?authuser=1

### Kane Egan-

Below are links to Kaggle notebooks used for work on the model:
https://www.kaggle.com/code/kaneegan/testloadimages
https://www.kaggle.com/code/kaneegan/offline-handwritten-text-ocr/edit
https://www.kaggle.com/code/kaneegan/testing
https://www.kaggle.com/code/kaneegan/handwriting-comparison-model
