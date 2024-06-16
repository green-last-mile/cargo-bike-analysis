import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

class ModelEvaluator:
    def __init__(self, predictions, actuals):
        """
        Initialize the evaluator with model predictions and actual values.
        :param predictions: List of tuples (hexagon_id, array of predicted quantiles [q05, q50, q95]).
        :param actuals: DataFrame containing actual values with columns ['h3', 'service_time'].
        """
        self.predictions = pd.DataFrame(predictions, columns=['hexagon', 'predicted_quantiles'])
        self.actuals = self.convert_to_dataframe(actuals)
        self.prepare_data()

    def convert_to_dataframe(self, actuals):
        """
        Convert actuals to a pandas DataFrame, preserving column names.
        """
        column_names = ['h3', 'service_time', 'city', 'service_time_log', 'h3_count']
        if not isinstance(actuals, pd.DataFrame):
            return pd.DataFrame(actuals, columns=column_names)
        return actuals

    def prepare_data(self):
        """
        Prepare the data for evaluation.
        """
        self.predictions[['q05', 'q50', 'q95']] = pd.DataFrame(self.predictions['predicted_quantiles'].tolist(), index=self.predictions.index)

    def pinball_loss(self, y_true, y_pred, quantile):
        """
        Calculate the Pinball Loss for a given quantile.
        """
        delta = y_true - y_pred
        return np.mean(np.maximum(quantile * delta, (quantile - 1) * delta))

    def calculate_coverage(self, actual, lower_bound, upper_bound):
        """
        Calculate the coverage probability between lower and upper quantiles.
        """
        return ((lower_bound <= actual) & (actual <= upper_bound)).mean()

    def evaluate_hex_level(self):
        """
        Evaluate the model at the hex level.
        to review.
        """
        hex_level_results = []
        for hex_id in self.predictions['hexagon'].unique():
            hex_pred = self.predictions[self.predictions['hexagon'] == hex_id]
            hex_actual = self.actuals[self.actuals['h3'] == hex_id]['service_time']

            # Count the number of deliveries in the hexagon
            num_deliveries = len(hex_actual)

            # Repeat the median (q50) prediction to match the length of actual values
            repeated_median_pred = np.full(hex_actual.shape, hex_pred['q50'].iloc[0])

            mae = mean_absolute_error(hex_actual, repeated_median_pred)
            rmse = mean_squared_error(hex_actual, repeated_median_pred, squared=False)

            pinball_loss_05 = self.pinball_loss(hex_actual, hex_pred['q05'].iloc[0], 0.05)
            pinball_loss_50 = self.pinball_loss(hex_actual, hex_pred['q50'].iloc[0], 0.50)
            pinball_loss_95 = self.pinball_loss(hex_actual, hex_pred['q95'].iloc[0], 0.95)

            coverage = self.calculate_coverage(hex_actual, hex_pred['q05'].iloc[0], hex_pred['q95'].iloc[0])

            hex_level_results.append({
                'hexagon': hex_id,
                'num_deliveries': num_deliveries,
                'mae': mae,
                'rmse': rmse,
                'pinball_loss_05': pinball_loss_05,
                'pinball_loss_50': pinball_loss_50,
                'pinball_loss_95': pinball_loss_95,
                'coverage': coverage
            })

        return pd.DataFrame(hex_level_results)

    def evaluate_city_level(self):
        """
        Evaluate the model at the city level.

        to review/complete.
        """


        # city_level_results = {}

        # # Aggregate the actual service times by hexagon to calculate empirical quantiles
        # actual_quantiles = self.actuals.groupby('h3')['service_time'].quantile([0.05, 0.50, 0.95]).unstack()

        # for quantile, q in zip(['q05', 'q50', 'q95'], [0.05, 0.50, 0.95]):
        #     # Extract the corresponding predicted quantile for each hexagon
        #     predicted_quantiles = self.predictions[['hexagon', quantile]]

        #     # Merge the actual and predicted quantiles
        #     merged_data = actual_quantiles.merge(predicted_quantiles, left_on='h3', right_on='hexagon')

        #     # Calculate MAE between empirical and predicted quantiles
        #     mae = mean_absolute_error(merged_data[q], merged_data[quantile])
        #     city_level_results[f'mae_{quantile}'] = mae

        #     # Calculate RMSE between empirical and predicted quantiles
        #     rmse = np.sqrt(mean_squared_error(merged_data[q], merged_data[quantile]))
        #     city_level_results[f'rmse_{quantile}'] = rmse

        # return city_level_results


    def evaluate(self):
        """
        Perform both hex-level and city-level evaluations.
        """
        hex_level_results = self.evaluate_hex_level()
        city_level_results = self.evaluate_city_level()
        return hex_level_results, city_level_results

# Example usage:
# evaluator = ModelEvaluator(model_predictions, actual_data)  # actual_data is a Pandas DataFrame
# hex_level_evaluation, city_level_evaluation = evaluator.evaluate()
