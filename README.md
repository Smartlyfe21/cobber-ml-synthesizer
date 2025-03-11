# CobberMLsyn: Predicting STALKiness with Machine Learning

## Project Overview
In an alternate universe called **Cobberland**, all matter is composed of substances known as **kernels**. Each kernel is characterized by three properties:
- **EARitability**
- **aMAIZEingness**
- **STALKiness** (a challenging property to quantify)

This project applies machine learning to predict **STALKiness** based on the more easily measurable properties **EARitability** and **aMAIZEingness**. Multiple ML models are explored to assess their predictive performance and to understand the trade-offs between model accuracy, computational efficiency, and overfitting.

## Tools and Workflow

- **Jupyter Notebook:**  
  Used for generating kernel data. The notebook `CreateNewKernels.ipynb` demonstrates how the kernel data was created.

- **PyCharm:**  
  The main analysis and GUI application were developed using PyCharm. The source code (now renamed to `CobberMLysn.py`) implements the Cobber Machine Learning Explorer.

This combination allowed for an efficient workflow, with Jupyter Notebook handling data generation and PyCharm providing robust tools for GUI development and model evaluation.

## Real-World Relevance
Machine learning is widely used in chemistry to predict molecular properties, design new compounds, and assist in drug discovery. This fictitious example demonstrates how regression analysis can be applied to predict properties, making it easier to translate these concepts into real-world applications.

## Machine Learning Models Implemented
1. **Linear Regression** – Simple and interpretable, but may not capture complex relationships.
2. **Decision Tree** – Captures non-linear relationships but can be prone to overfitting.
3. **Random Forest** – Uses an ensemble of decision trees to improve prediction accuracy and reduce overfitting.
4. **Support Vector Machine (SVM)** – Effective in high-dimensional spaces but requires significant computational resources.
5. **k-Nearest Neighbors (k-NN)** – Predicts STALKiness based on the values of nearby kernels; performance depends on the choice of *k*.

## Data Splitting Strategy
The dataset is divided into two subsets:
- **Training Set:** Used to train the model and learn underlying patterns.
- **Test Set:** Used to evaluate the model’s ability to generalize to unseen data.

This split helps prevent overfitting and ensures that the model's performance is evaluated on new, unseen data.

## Running the Project
1. **Load the Dataset:**  
   Use the GUI to load `KernelData.csv`, which contains the kernel data with columns for Kernel ID, EARitability, aMAIZEingness, and STALKiness.
2. **Select a Machine Learning Model:**  
   Choose one of the following models from the dropdown:  
   - Linear Regression  
   - Decision Tree  
   - Random Forest  
   - Support Vector Machine  
   - k-Nearest Neighbors
3. **Set the Train/Test Split:**  
   Use the slider to select the percentage split (default is 80% for training and 20% for testing).
4. **Train the Model:**  
   Click "Run ML Training" to train the selected model. Performance metrics (MSE, MAE, and R²) and two plots (Actual vs. Predicted and Residuals) will be displayed.
5. **Make Predictions:**  
   Enter values for EARitability and aMAIZEingness to predict STALKiness.
6. **Save Outputs:**  
   Use the "Save All Plots" button to save the Actual vs. Predicted and Residuals plots as PNG files, and the "Save Predictions" button to export predictions and performance metrics to a CSV file.

## Experimental Results

After training and evaluating the models, the following results were observed:

| **Model**                  | **Training Time (s)** | **MSE**    | **MAE**    | **R²**    | **Predicted STOCKiness** |
|----------------------------|-----------------------|------------|------------|-----------|--------------------------|
| **Linear Regression**      | 0.0129                | 133.7344   | 8.9180     | 0.8873    | 34.4249                  |
| **Decision Tree**          | 0.1471                | 1.5361     | 1.0111     | 0.9987    | 5.7641                   |
| **Random Forest**          | 6.9481                | 0.9260     | 0.8096     | 0.9992    | 7.2871                   |
| **Support Vector Machine** | 29.7454               | 15.0352    | 2.2255     | 0.9873    | 23.2152                  |
| **k-Nearest Neighbors**    | 0.0234                | 1.8629     | 0.8802     | 0.9984    | 22.4809                  |

### Key Observations
- **Linear Regression** produced a moderate R² with relatively high error metrics, suggesting that a simple linear model may not capture the full complexity of the data.
- **Decision Tree** and **k-NN** achieved high R² values with low error metrics, indicating strong predictive performance but potentially sensitive to data splits.
- **Random Forest** provided excellent performance with the lowest errors and highest R², balancing bias and variance effectively.
- **Support Vector Machine** delivered good results but required significantly more training time.

## Conclusion
This project demonstrates that various machine learning models can effectively predict the complex property (**STALKiness**) of kernels using **EARitability** and **aMAIZEingness** as inputs. The analysis highlights trade-offs among models regarding training time, prediction accuracy, and computational cost, providing valuable insights for real-world applications in chemistry and materials science.

## Acknowledgements

This project builds upon ideas and methodologies obtained from online resources and guidance provided by [Prof. Darin](https://www.darinulness.com/teaching/machine-learning-for-undergraduate-chemistry). We are grateful for his contributions and support.

## Contact and Support

For more information, support, or to provide feedback, please visit:
- **Website:** [www.darinulness.com](https://www.darinulness.com)
- **Concordia College Chemistry Department:** Visit the department page for additional resources and contact information.


---

For more details on the methodology and experiments, refer to the **Cobber Machine Learning Explorer Guide** included in this repository.
