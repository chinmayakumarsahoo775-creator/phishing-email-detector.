Phishing Email Detection Model

A beginner-friendly machine learning project that classifies email text as Phishing or Safe using Python and Scikit-learn.

Features

- Trains a machine learning model on labeled email examples.
- Uses TF-IDF to analyze email text.
- Checks basic URL counts and suspicious keyword patterns.
- Predicts whether new email text may be phishing.
- Displays accuracy, a classification report, and a confusion matrix.

Requirements

- Python 3.10 or newer
- pandas
- scikit-learn
- matplotlib

Installation

1. Download this repository using Code → Download ZIP.

2. Extract the ZIP file.

3. Open a terminal in the extracted folder.

4. Install the required libraries:
   
   python -m pip install -r requirements.txt

5. Run the project:
   
   python phishing_detector.py

6. Enter the email text you want to test when prompted.

Project Files

- "phishing_detector.py" — model training, evaluation, and prediction.
- "emails.csv" — sample labeled email dataset.
- "requirements.txt" — required Python packages.
- "README.md" — project documentation.

Machine Learning

The project uses TF-IDF text features and Logistic Regression.

Limitations

The included dataset is small and intended for demonstration. Its accuracy does not represent real-world performance. A larger, diverse dataset and independent testing are required before practical deployment.

Responsible Use

This tool is an educational aid. A prediction of Safe does not guarantee that an email is legitimate. Do not enter real passwords, confidential emails, or personal information.

Technologies Used

Python, pandas, Scikit-learn, TF-IDF, Logistic Regression, and Matplotlib. 
