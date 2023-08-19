import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

class preprocessing():
    def __init__(self, Dataset) -> None:
        self.dataset = Dataset
    
    def convert_date_to_datetime(self, column:str):
        """
        convert date column to datetime
        """
        self.dataset[column] = pd.to_datetime(self.dataset[column])
        return self.dataset
    
    def fix_dataset_index(self):
        """
        fix dataset index
        """
        self.dataset = self.dataset.reset_index(drop=True)
        return self.dataset
    
    def remove_data_values(self, threshold:str):
        """
        remove data values less than threshold
        """
        self.dataset = self.dataset[self.dataset['date'] > threshold]
        return self.dataset
    
    def check_missing_values(self):
        print(self.dataset.isnull().sum())
        
    def drop_columns(self, columns:list):
        """
        drops column in columns list from dataset
        """
        self.dataset = self.dataset.drop(columns, axis=1)
        return self.dataset
    
    def perform_temporal_imputation(self, column:str):
        """
        Temporal imputation for missing Attendance using previous 3 matches
        """
        for index, row in self.dataset.iterrows():
            if pd.isnull(row[column]) and row["venue"] == "Home":
                home_team = row["team"]
                match_date = row["date"]
                
                previous_matches = self.dataset[(self.dataset["team"] == home_team) & (self.dataset["date"] < match_date)]
                previous_matches = previous_matches.nlargest(3, "date")

                if not previous_matches.empty:
                    avg_attendance = previous_matches[column].mean()
                    self.dataset.at[index, column] = avg_attendance
                    
        return self.dataset

    
    def perform_label_column_encoding(self, column:str):
        """
        label encoding for categoical columns
        """
        # Label encoding for 'result'
        label_encoder = LabelEncoder()
        for col in column:
            self.dataset[col] = label_encoder.fit_transform(self.dataset[col])

        return self.dataset
    
    def plot_distributions(self, column:str):
        """
        plot distributions for numerical columns to ascertain the best normalization technique to use
        """
        # Plot histograms to visualize distributions
        plt.figure(figsize=(12, 8))
        self.dataset[column].hist(bins=20, figsize=(12, 8))
        plt.tight_layout()
        plt.show()
        
    def perform_normalization(self, column:str, type_:str):
        """
        normalization for numerical columns
        """
        # Standardization
        scaler = StandardScaler() if type_ == "standard" else MinMaxScaler()

        for col in column:
            self.dataset[col] = scaler.fit_transform(self.dataset[col].values.reshape(-1, 1))
        return self.dataset

    def split_dataset(self,features:list, column:str="date", target:str="result"):
        """
        split dataset into train and test based on time while also ensuring 
        our train dataset accounts for 80 percent of the total dataset by 
        automatically calculating the cutoff date
        """
        
        # sort the data by date in ascending order
        self.dataset = self.dataset.sort_values(by=column)
        
        # calculate total number of samples
        total_samples = len(self.dataset)
        
        # calculate the index for approximately 80% training data
        train_samples = int(0.8 * total_samples)
        
        # get the cutoff date corresponding to the calculated index
        cutoff_date = self.dataset.iloc[train_samples][column]
        
        # Split the dataset into train and test sets
        train_data = self.dataset[self.dataset[column] < cutoff_date]
        test_data = self.dataset[self.dataset[column] >= cutoff_date]
        
        X_train = train_data[features]
        y_train = train_data[target]
        X_test = test_data[features]
        y_test = test_data[target]
        
        return X_train, y_train, X_test, y_test
    
    