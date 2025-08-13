import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import joblib
import numpy as np
from utils.logger import get_logger
import io

logger = get_logger("evaluation")
buffer = io.StringIO()
processed_columns = [
        'Hour','Day','Boundary','Suspicious_car_rental','Suspicious_fuel',
        'Cumulative_type_percent','Cumulative_Unique_Locations','Days_since_last'
    ]

def evaluate_dataset(enhanced_path):
    try:
        os.makedirs("evaluation_outputs", exist_ok=True)
        information = pd.read_csv(enhanced_path)
        information.info(buf=buffer)
        logger.info("Dataset Info:\n" + buffer.getvalue())
        logger.info("\nSummary Statistics:\n%s", information.describe().to_string())
        logger.info("\nMissing Values:\n%s", information.isnull().sum().to_string())

        plt.figure(figsize=(8, 6))
        sns.heatmap(information[processed_columns].corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.savefig(os.path.join("evaluation_outputs", "correlation_heatmap.png"))
        plt.close()
    except Exception as e:
        logger.error(f'Could not evaluate the dataset or produce a correleation-heatmap: {(e)}')    


def evaluate_model(df_path, model_path, output_dir="evaluation_outputs"):
    try:

        os.makedirs(output_dir, exist_ok=True)

        df = pd.read_csv(df_path)
        model_data = joblib.load(model_path)
        isolation_forest = model_data['isolation_forest']
        scaler = model_data['scaler']

        features = df[processed_columns]

        sample_data = features.sample(n=500, random_state=40)
        sample_scaled = scaler.transform(sample_data)

        explainer = shap.Explainer(isolation_forest.decision_function, sample_scaled)
        shap_values = explainer(sample_scaled)
        logger.info("producing shap summary plot")
        shap.summary_plot(shap_values, sample_scaled, feature_names=processed_columns, show=False)
        plt.savefig(os.path.join(output_dir, "shap_summary_iforest.png"))
        plt.close()

        if 'Both_prediction' in df.columns:
            count = df['Both_prediction'].sum()
            logger.info("Detected %d fraudulent transactions out of %d", count, len(df))



        anomalous_customers = df[df['Forest_prediction'] == 1]['Customer_id'].unique()
        anomalous_sample = df[df['Forest_prediction'] == 1].sample(n=1, random_state=40)
        anomalous_sample_features = anomalous_sample[processed_columns]
        anomalous_sample_scaled = scaler.transform(anomalous_sample_features)

        explainer2 = shap.Explainer(isolation_forest.decision_function, anomalous_sample_scaled)
        shap_values_single = explainer2(anomalous_sample_scaled)
        if len(anomalous_customers) > 0:
            chosen_customer = np.random.choice(anomalous_customers)
            customer_df = df[df['Customer_id'] == chosen_customer]
            logger.info(f"SHAP waterfall plot for a suspicious transaction by {chosen_customer}")
            shap.plots.waterfall(shap_values_single[0], show=False)
            plt.savefig(os.path.join(output_dir, "shap_waterfall_anomalous.png"))
            plt.close()

            if 'Transaction_amount' in customer_df.columns:
                logger.info("Producing scatter plot")
                plt.figure(figsize=(8, 6))
                sns.scatterplot(
                    data=customer_df,
                    x='Hour',
                    y='Transaction_amount',
                    hue='Forest_prediction',
                    palette={0: 'green', 1: 'red'},
                    alpha=0.7
                )
                plt.title(f"Transactions for {chosen_customer}")
                plt.xlabel("Hour of Day")
                plt.ylabel("Transaction Amount")
                plt.legend(title="Anomaly")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"transactions_{chosen_customer}.png"))
                plt.close()    
    except Exception as e:
        logger.error(f'Could not generate plots: {(e)}')            