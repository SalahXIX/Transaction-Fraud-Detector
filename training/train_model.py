import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import joblib
from utils.logger import get_logger
import mlflow

logger = get_logger("training")

class FraudEnsembleModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.components = joblib.load(context.artifacts["model_file"])
        self.scaler = self.components['scaler']
        self.isolation_forest = self.components['isolation_forest']
        self.lof = self.components['lof']
        self.threshold_iso = self.components['threshold_iso']
        self.threshold_lof = self.components['threshold_lof']

    def predict(self, context, model_input: pd.DataFrame):
        scaled = self.scaler.transform(model_input)
        iso_score = self.isolation_forest.decision_function(scaled)
        iso_pred = (iso_score < self.threshold_iso).astype(int)
        lof_score = self.lof.decision_function(scaled)
        lof_pred = (lof_score < self.threshold_lof).astype(int)
        return ((iso_pred + lof_pred) >= 1).astype(int)
    
def train_and_save(df, model_path, rate=0.02):
    try:
        logger.info(f"Training models with contamination rate: {rate}")
        processed_columns = [
            'Hour','Day','Boundary','Suspicious_car_rental','Suspicious_fuel',
            'Cumulative_type_percent','Cumulative_Unique_Locations','Days_since_last'
        ]

        features = df[processed_columns]
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)

        with mlflow.start_run():
            mlflow.log_param("contamination_rate", rate)

            isolation_forest = IsolationForest(n_estimators=100, contamination=rate, random_state=40)
            isolation_forest.fit(scaled)
            score = isolation_forest.decision_function(scaled)
            threshold_iso = pd.Series(score).quantile(rate)
            df['Forest_prediction'] = (score < threshold_iso).astype(int)

            lof = LocalOutlierFactor(n_neighbors=65, contamination=rate, novelty=True)
            lof.fit(scaled)
            lscore = lof.decision_function(scaled)
            threshold_lof = pd.Series(lscore).quantile(rate)
            df['LOF_prediction'] = (lscore < threshold_lof).astype(int)
            df['Both_prediction'] = ((df['Forest_prediction'] + df['LOF_prediction']) >= 1).astype(int)
            df.to_csv("transactions_enhanced.csv", index=False)

            fraud_count = df['Both_prediction'].sum()
            mlflow.log_metric("num_frauds_detected", int(fraud_count))

            logger.info("Saving models and thresholds")
            joblib.dump({
                'isolation_forest': isolation_forest,
                'lof': lof,
                'scaler': scaler,
                'threshold_iso': threshold_iso,
                'threshold_lof': threshold_lof
            }, model_path)

            logger.info("Logging ensemble model to MLflow")
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=FraudEnsembleModel(),
                artifacts={"model_file": model_path},
                registered_model_name="fraud_detection_model"
            )
            logger.info("Model training, saving, and registration complete.")
            return df
    except Exception as e:
        logger.error(f'Failed to train or save model: {(e)}')
   
