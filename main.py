import os
from data.load_data import GenerateData
from utils.preprocessing import preprocess_data
from training.train_model import train_and_save
from evaluation.evaluate import evaluate_dataset, evaluate_model
from utils.logger import get_logger

logger = get_logger("Main")

if __name__ == "__main__":
    try:
        base_path = os.getcwd()
        os.makedirs(os.path.join(base_path, "models"), exist_ok=True)
        cleaned = os.path.join(base_path, "cleaned_transactions.csv")
        enhanced = os.path.join(base_path, "transactions_enhanced.csv")
        model_path = os.path.join(base_path, "models", "EnsembleLearning.pkl")

        GenerateData(cleaned)
        df = preprocess_data(cleaned, enhanced)
        df = train_and_save(df, model_path)
        evaluate_dataset(enhanced)
        evaluate_model(enhanced, model_path)
    except Exception as e:
        logger.error(f'Main file error: {(e)}')       
