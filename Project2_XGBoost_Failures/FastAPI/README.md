To get the application up and running, follow these steps:

1.  **Open your terminal** and navigate to the project directory.
2.  **Run the Uvicorn server** by executing the following command:
    ```bash
    python -m uvicorn fast_API_ML:app --reload
    ```

## Usage Instructions

Once the server starts:

* **Access the URL:** A link with a port number (e.g., `http://127.0.0.1:8000`) will be displayed in the terminal. **Left-click** and follow the link.
* **Open Swagger UI:** To interact with the ML model, add `/docs` at the end of the URL in your browser address bar:
    > Example: `http://127.0.0.1:8000/docs`
* **Test the Model:** From the Swagger UI, you can input parameters and test the predictions of the ML model directly.
