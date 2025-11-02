
# Handwriting Recognition API

This project provides a web API for handwriting recognition. It uses a CRNN model trained on the IAM Handwriting Database.

## Local Setup

1.  **Clone the repository:**
    ```
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Install dependencies:**
    ```
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```
    uvicorn app:app --reload
    ```

4.  **Access the API:**
    The API will be available at `http://localhost:8000`.

## Docker

1.  **Build the Docker image:**
    ```
    docker build -t handwriting-recognition-app .
    ```

2.  **Run the Docker container:**
    ```
    docker run -p 8000:8000 handwriting-recognition-app
    ```

## Heroku Deployment

Before deploying to Heroku, you need to modify the `app.py` file to use the environment variable for the model path. 

1.  **Comment out the line that uses the hardcoded path:**
    ```python
    # configs = BaseModelConfigs.load("Models/04_sentence_recognition/202510280421/configs.yaml")
    ```

2.  **Uncomment the line that uses the environment variable:**
    ```python
    # model_path = os.environ.get("MODEL_PATH", "Models/04_sentence_recognition/202510280421/configs.yaml")
    # configs = BaseModelConfigs.load(model_path)
    ```

Once you have made these changes, you can proceed with the Heroku deployment.

1.  **Create a Heroku account and install the Heroku CLI.**

2.  **Login to Heroku:**
    ```
    heroku login
    ```

3.  **Create a Heroku app:**
    ```
    heroku create <your-app-name>
    ```

4.  **Set the model path environment variable:**
    ```
    heroku config:set MODEL_PATH="Models/04_sentence_recognition/202510280421/configs.yaml" -a <your-app-name>
    ```

5.  **Push to Heroku:**
    ```
    git push heroku master
    ```

## API Endpoints

*   `GET /`: Returns a welcome message.
*   `POST /predict/`: Accepts an image file and returns the predicted text.

