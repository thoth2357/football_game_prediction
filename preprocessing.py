class preprocessing():
    def __init__(self, Dataset) -> None:
        self.dataset = Dataset
        
    def check_missing_values(self):
        print(self.dataset.isnull().sum())
    
    