import pandas as pd
import numpy as np
from faker import Faker
import random
from utils.logger import get_logger

logger = get_logger("loading")

    
def boundary(amount):
    if amount <= 500:
        return 0
    if amount <= 1000:
        return 1
    else:
        return 2
    
def GenerateData(saved_path, total=80000):
    try:
        fake = Faker()
        np.random.seed(42)
        data = []
        Locations = ['USA', 'UK','INDIA', 'PAKISTAN', 'BANGLADESH']
        Merchant_categories= ['DINING', 'ELECTRONICS', 'FUEL', 'GROCERIES', 'CARS']
        Transaction_types = ["ATM", "ONLINE", "POS"]


        for i in range(total):
            Transaction_id = f'Transaction{i:05d}'
            Customer_id = 'Customer' + str(f'{random.randint(0, 15000):03d}')
            Transaction_date = fake.date_time_between(start_date='-1y', end_date='now')  
            Transaction_amount = round(np.random.exponential(scale=300), 2)
            Merchant_category = random.choice(Merchant_categories)
            Location = random.choice(Locations)
            Transaction_type = random.choice(Transaction_types)                      
            data.append([Transaction_id, Customer_id, Transaction_date, Transaction_amount, Merchant_category, Location, Transaction_type])        


        df = pd.DataFrame(data, columns=['Transaction_id','Customer_id','Transaction_date','Transaction_amount','Merchant_category','Location','Transaction_type'])
        df['Location'] = df['Location'].str.upper().str.strip()
        df['Merchant_category'] = df['Merchant_category'].str.title().str.strip().str.upper()
        df['Transaction_date'] = pd.to_datetime(df['Transaction_date'])
        df = df[df['Transaction_amount'] < 2500]
        df = df.drop_duplicates(subset=df.columns[1:])

        df.to_csv(saved_path, index=False)
        logger.info("Generated and saved data sucessfully")
        return saved_path
    except Exception as e:
        logger.error(f'Error in generating and saving data: {(e)}')    