# Use a base image with Python and necessary libraries
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port that Streamlit will run on
EXPOSE 8501

# Define the command to run the application
CMD ["streamlit", "run", "streamlit_app.py"]