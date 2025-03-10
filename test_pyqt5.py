import sys
import time
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel, QComboBox,
    QSlider, QLineEdit, QVBoxLayout, QHBoxLayout, QFileDialog, QGridLayout, QTextEdit, QGroupBox, QScrollArea
)
from PyQt5.QtCore import Qt
import pyqtgraph as pg
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr


class MLGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cobber Machine Learning Explorer")
        self.setGeometry(100, 100, 1200, 800)  # Increased size for better layout

        # Initialize data and model
        self.data = None
        self.model = None

        # Set up the main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QGridLayout()
        self.main_widget.setLayout(self.layout)

        # Create and add widgets
        self.create_widgets()

    def create_widgets(self):
        # Left Panel for Controls
        self.control_group = QGroupBox("Controls")
        self.control_layout = QVBoxLayout()
        self.control_group.setLayout(self.control_layout)

        # Open CSV Button
        self.open_button = QPushButton("Open CSV")
        self.open_button.clicked.connect(self.open_file)
        self.control_layout.addWidget(self.open_button)

        # ML Algorithm Selection
        self.alg_label = QLabel("Select ML Algorithm")
        self.control_layout.addWidget(self.alg_label)

        self.algorithms = [
            "Linear Regression", "Decision Tree", "Random Forest",
            "Support Vector Machine", "k-Nearest Neighbors"
        ]
        self.alg_combo = QComboBox()
        self.alg_combo.addItems(self.algorithms)
        self.control_layout.addWidget(self.alg_combo)

        # Train/Test Split Slider
        self.split_label = QLabel("Train/Test Split (%)")
        self.control_layout.addWidget(self.split_label)

        self.split_slider = QSlider(Qt.Horizontal)
        self.split_slider.setMinimum(50)
        self.split_slider.setMaximum(90)
        self.split_slider.setValue(80)
        self.split_slider.setTickInterval(5)
        self.split_slider.setTickPosition(QSlider.TicksBelow)
        self.control_layout.addWidget(self.split_slider)

        # Save Plot Button
        self.save_button = QPushButton("Save Plot")
        self.save_button.clicked.connect(self.save_plot)
        self.control_layout.addWidget(self.save_button)

        # Display current split percentage
        self.split_value_label = QLabel("80% Train / 20% Test")
        self.control_layout.addWidget(self.split_value_label)
        self.split_slider.valueChanged.connect(self.update_split_label)

        # Run Training Button
        self.train_button = QPushButton("Run ML Training")
        self.train_button.clicked.connect(self.run_training)
        self.control_layout.addWidget(self.train_button)

        # Prediction Inputs
        self.pred_label = QLabel("Predict STOCKiness")
        self.control_layout.addWidget(self.pred_label)

        self.x1_entry = QLineEdit()
        self.x1_entry.setPlaceholderText("Enter EARitability")
        self.control_layout.addWidget(self.x1_entry)

        self.x2_entry = QLineEdit()
        self.x2_entry.setPlaceholderText("Enter aMAIZEingness")
        self.control_layout.addWidget(self.x2_entry)

        self.pred_button = QPushButton("Predict STOCKiness")
        self.pred_button.clicked.connect(self.predict_y)
        self.control_layout.addWidget(self.pred_button)

        # Spacer
        self.control_layout.addStretch()

        # Add Control Panel to the Main Layout
        self.layout.addWidget(self.control_group, 0, 0, 1, 1)

        # Right Panel for Graphs and Console
        self.display_group = QGroupBox("Display")
        self.display_layout = QVBoxLayout()
        self.display_group.setLayout(self.display_layout)

        # Plot Layout
        self.plot_layout = QHBoxLayout()

        # Actual vs Predicted Plot
        self.graph_widget1 = pg.PlotWidget(title="Actual vs Predicted STOCKiness")
        self.graph_widget1.setBackground('k')
        self.graph_widget1.showGrid(x=True, y=True)
        self.graph_widget1.setLabel('left', 'Predicted STOCKiness')
        self.graph_widget1.setLabel('bottom', 'Actual STOCKiness')
        self.plot_layout.addWidget(self.graph_widget1)

        # Residuals Plot
        self.graph_widget2 = pg.PlotWidget(title="Residuals Plot")
        self.graph_widget2.setBackground('k')
        self.graph_widget2.showGrid(x=True, y=True)
        self.graph_widget2.setLabel('left', 'Residuals')
        self.graph_widget2.setLabel('bottom', 'Actual STOCKiness')
        self.plot_layout.addWidget(self.graph_widget2)

        self.display_layout.addLayout(self.plot_layout)

        # Embedded Console
        self.console_group = QGroupBox("Console Output")
        self.console_layout = QVBoxLayout()
        self.console_group.setLayout(self.console_layout)

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet("background-color: black; color: white;")
        self.console_layout.addWidget(self.console)

        self.display_layout.addWidget(self.console_group)

        # Add Display Panel to the Main Layout
        self.layout.addWidget(self.display_group, 0, 1, 1, 3)

    def update_split_label(self, value):
        test_percentage = 100 - value
        self.split_value_label.setText(f"{value}% Train / {test_percentage}% Test")

    def append_console(self, message):
        self.console.append(message)

    def open_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options
        )
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                self.append_console("CSV file loaded successfully!")
                self.append_console(f"Data Shape: {self.data.shape}\n")
            except Exception as e:
                self.append_console(f"Failed to load file: {str(e)}\n")

    def run_training(self):
        if self.data is not None:
            try:
                X = self.data[['EARitability', 'aMAIZEingness']].values
                Y = self.data['STOCKiness'].values
            except KeyError as e:
                self.append_console(f"Missing column in data: {e}\n")
                return

            train_size = self.split_slider.value() / 100
            X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, train_size=train_size, random_state=42
            )

            # Select and train the model based on the user's choice
            alg = self.alg_combo.currentText()
            if alg == "Linear Regression":
                self.model = LinearRegression()
            elif alg == "Decision Tree":
                self.model = DecisionTreeRegressor(random_state=42)
            elif alg == "Random Forest":
                self.model = RandomForestRegressor(random_state=42)
            elif alg == "Support Vector Machine":
                self.model = SVR()
            elif alg == "k-Nearest Neighbors":
                self.model = KNeighborsRegressor()
            else:
                self.append_console("Unknown algorithm selected.\n")
                return

            # Record the start time
            start_time = time.time()

            # Train the selected model
            self.model.fit(X_train, Y_train)

            # Record the end time and calculate training duration
            training_time = time.time() - start_time

            # Make predictions
            Y_pred = self.model.predict(X_test)

            # Calculate performance metrics
            mse = mean_squared_error(Y_test, Y_pred)
            mae = mean_absolute_error(Y_test, Y_pred)
            r2 = r2_score(Y_test, Y_pred)

            # Print the results to the embedded console
            self.append_console("=== Training Complete ===")
            self.append_console(f"Algorithm: {alg}")
            self.append_console(f"Training Time: {training_time:.4f} seconds")
            self.append_console(f"MSE: {mse:.4f}")
            self.append_console(f"MAE: {mae:.4f}")
            self.append_console(f"R²: {r2:.4f}\n")

            # Plot Actual vs Predicted
            self.graph_widget1.clear()
            self.graph_widget1.plot(Y_test, Y_pred, pen=None, symbol='o', symbolBrush='b',symbolSize=3)

            # Correctly plot the 45-degree line
            min_val = min(Y_test.min(), Y_pred.min())
            max_val = max(Y_test.max(), Y_pred.max())
            self.graph_widget1.plot([min_val, max_val], [min_val, max_val], pen=pg.mkPen('r', width=2, style=Qt.DashLine))

            self.graph_widget1.setXRange(min_val, max_val)
            self.graph_widget1.setYRange(min_val, max_val)

            # Plot Residuals
            residuals = Y_test - Y_pred
            self.graph_widget2.clear()
            self.graph_widget2.plot(Y_test, residuals, pen=None, symbol='o', symbolBrush='g',symbolSize=3)
            self.graph_widget2.plot([min(Y_test), max(Y_test)], [0, 0], pen=pg.mkPen('r', width=2, style=Qt.DashLine))
            self.graph_widget2.setXRange(min_val, max_val)

            # Save predictions and metrics
            self.save_predictions(Y_test, Y_pred, mse, mae, r2, alg)


        else:
            self.append_console("No data loaded. Please load a CSV file first.\n")

    def predict_y(self):
        if self.model is not None:
            try:
                x1 = float(self.x1_entry.text())
                x2 = float(self.x2_entry.text())
                y_pred = self.model.predict([[x1, x2]])
                self.append_console(f"Predicted STOCKiness: {y_pred[0]:.4f}\n")
            except ValueError:
                self.append_console("Invalid input. Please enter valid numbers for EARitability and aMAIZEingness.\n")
            except Exception as e:
                self.append_console(f"Prediction failed: {str(e)}\n")
        else:
            self.append_console("No model trained. Please train a model first.\n")

    def save_predictions(self, Y_test, Y_pred, mse, mae, r2, algorithm):
        # Create a DataFrame
        results_df = pd.DataFrame({
            'Actual STOCKiness': Y_test,
            'Predicted STOCKiness': Y_pred,
            'MSE': [mse] * len(Y_test),
            'MAE': [mae] * len(Y_test),
            'R²': [r2] * len(Y_test)
        })

        # Open a file dialog to choose the save location and filename
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, f"Save {algorithm} Predictions", "", "CSV Files (*.csv);;All Files (*)", options=options
        )
        if file_path:
            # Ensure the file has the correct extension
            if not file_path.lower().endswith('.csv'):
                file_path += '.csv'

            # Save the DataFrame to a CSV file
            results_df.to_csv(file_path, index=False)
            self.append_console(f"{algorithm} predictions and metrics saved to {file_path}\n")

    def save_plot(self):
        if self.graph_widget1.isVisible():
            # Open a file dialog to choose the save location and filename
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Plot", "", "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)", options=options
            )
            if file_path:
                # Ensure the file has the correct extension
                if not file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path += '.png'

                # Export the plot to the chosen file
                exporter = pg.exporters.ImageExporter(self.graph_widget1.plotItem)
                exporter.parameters()['width'] = 800  # Set the width of the saved image
                exporter.export(file_path)
                self.append_console(f"Plot saved to {file_path}\n")
        else:
            self.append_console("No plot available to save.\n")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MLGui()
    window.show()
    sys.exit(app.exec_())
