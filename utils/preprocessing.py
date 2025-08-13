import pandas as pd
from data.load_data import boundary
from utils.logger import get_logger

logger = get_logger("preprocessing")


def preprocess_data(cleaned_path,enhanced_path):
    try:
        df = pd.read_csv(cleaned_path)
        df['Transaction_date'] = pd.to_datetime(df['Transaction_date'])
        df = df.sort_values(by=['Customer_id','Transaction_date'])
        df['Boundary'] = df['Transaction_amount'].apply(boundary)
        df['Hour'] = df['Transaction_date'].dt.hour
        df['Day'] = df['Transaction_date'].dt.weekday + 2 
        df['Day'] = df['Day'].replace({8:1})
        df['Suspicious_car_rental'] = ((df['Merchant_category'] == 'CARS') & (df['Transaction_amount'] < 500)).astype(int)
        df['Suspicious_fuel'] = ((df['Merchant_category'] == 'FUEL') & (df['Transaction_amount'] > 300)).astype(int)
        df['Days_since_last'] = df.groupby('Customer_id')['Transaction_date'].diff().dt.days.fillna(0)

        TransactionsPerCustomer = df.groupby('Customer_id').cumcount() + 1
        TypeCount = df.groupby(['Customer_id','Transaction_type']).cumcount() + 1
        df['Cumulative_type_percent'] = round(TypeCount / TransactionsPerCustomer, 2)
        df['Cumulative_Unique_Locations'] = df.assign(new_location=~df.duplicated(subset=['Customer_id','Location'])).groupby('Customer_id')['new_location'].cumsum()

        df = df.drop(columns=['Transaction_id', 'Transaction_date', 'Merchant_category', 'Location', 'Transaction_type'])

        df.to_csv(enhanced_path, index=False)
        logger.info("CSV file saved sucessfully")
        return df
    except Exception as e:
        logger.error(f'Preprocessing failed: {(e)}')      