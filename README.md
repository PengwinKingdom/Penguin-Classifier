
<h1>Penguin Species Classifier</h1>
<p>A web-based image classifier to identify penguin species using a trained deep learning model.</p>

<h2>Project Overview</h2>
<p>This project allows users to upload a photo of a penguin and receive a prediction of its species. The prediction is powered by a convolutional neural network (CNN) trained on images from 18 different species. </p>

<h2>How It Works</h2>
<ul>
  <li>User uploads an image through the web interface</li>
  <li>The image is preprocessed</li>
  <li>The trained modelpredicts the species</li>
  <li>The predicted species is returned and displayed in the browser/li>
</ul>

<h2>Features</h2>
<ul>
  <li>Supports classification of 18 different penguin species</li>
  <li>Uses MobileNetV2 with transfer learning </li>
  <li>Simple and clean web interface built with Flask, HTML and CSS</li>
  <li>Real-time prediction directly from uploaded images</li>
</ul>

## Project Structure
```bash
  project/
├── model/
│   ├── class_indices.json
│   ├── penguin_classifier.h5
├── static/
│   ├── css/
│   │   ├── result.css
│   │   ├── style.css
│   ├── js/
│   │   ├── script.js
├── templates/
│   ├── index.html
│   ├── result.html
├── train_model.py
├── app.py
└── train_model.py
```
⚠️ Note: The file penguin_classifier.h5 is not included in the repository due to its large size.
It will be downloaded from Google Drive when running the application using the MODEL_URL environment variable.

<h2>Topics I Learned</h2>
<ul>
  <li>Preprocessing and normalizing image datasets for classification tasks</li>
  <li>Fine-tuning pretrained CNN models with transfer learning</li>
  <li>Building a functional web app with Flask and integrating ML models</li>
  <li>Creating user-friendly frontends with HTML and CSS</li>
</ul>

## Installation & Usage

## Prerequisites

- **Python 3.8+** must be installed on your system
    - To check if Python is installed, run:
      ```bash
       python --version
      ```
- **pip** (Python package manager) should be available
  - To check if pip is installed, run:
      ```bash
       pip --version
      ```
- **Flask and other dependencies** will be installed automatically using the command:
  ```bash
  pip install -r requirements.txt
  ```

## 1. Clone the repository
   ```bash
   git clone https://github.com/PengwinKingdom/Penguin-Classifier.git
   cd Penguin-Classifier
   ```

## 2. Create and activate a virtual environment
   **Mac/Linux**
   ```bash
   python -m venv env
   source env/bin/activate
   ```
   **Windows**
   ```bash
   python -m venv env
   .\env\Scripts\activate
   ```
   
## 3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
   
## 4. Set the model URL
   **Mac/Linux**
   ```bash
   export MODEL_URL="https://drive.google.com/uc?id=1q4kJQPJf0-0jRyv2wfYV7XCjMxYLX-OT"
   ```
   **Windows**
   ```bash
   setx MODEL_URL "https://drive.google.com/uc?id=1q4kJQPJf0-0jRyv2wfYV7XCjMxYLX-OT"
   ```
   
## 5. Run the application
   **Mac/Linux**
   ```bash
   export FLASK_APP=app.py
   export FLASK_ENV=development
   flask run
   ```
   **Windows**
   ```bash
   $env:FLASK_APP="app.py"
   $env:FLASK_ENV="development"
   flask run
   ```

## 6. Open your browser at:
   http://127.0.0.1:5000
   
   
