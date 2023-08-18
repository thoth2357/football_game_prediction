import pandas as pd
class preprocessing():
    def __init__(self, Dataset) -> None:
        self.dataset = Dataset
    
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
    
    
    
    
    
    
    