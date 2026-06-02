import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import List, Dict, Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO,format="%(asctime)s-%(name)s-%(levelname)s-%(message)s")

logger = logging.getLogger(__name__)


class AmazonDataAnalyzer:
    
    def __init__(self,data_path:str):
        self.data_path = Path(data_path)
        self.df: Optional[pd.DataFrame] = None
        sns.set_theme(style="whitegrid")
        
    def load_dataset(self,nrows:Optional[int]=None)->None:
        logger.info(f"loading data from {self.data_path}") 
        
        try:
            self.df = pd.read_json(self.data_path,lines=True,nrows=nrows,chunksize=100000)   
            
            if isinstance(self.df,pd.io.json._json.JsonReader):
                self.df = pd.concat(self.df,ignore_index=True)
            logger.info(f"Data loaded successfully and the Shape of dataset is {self.df.shape}")
            self._optimize_memory()
        
        except Exception as e:
            logger.error(f"Failed to load data {e}")
            raise
        
    def _optimize_memory(self)->None:
        if self.df is None: return
        
        start_mem = self.df.memory_usage().sum()/1024**2
        
        for col in self.df.columns:
            if self.df[col].dtype == 'float64':
                self.df[col] = self.df[col].astype('float32')
            elif self.df[col].dtype == 'int64':
                self.df[col] = self.df[col].astype('int32')
        
        end_mem = self.df.memory_usage().sum()//1024**2
        
        logger.info(f"Memory optimized: {start_mem:.2f}MB -> {end_mem:.2f}MB")
        logger.info(f"Memory optmized by {((end_mem-start_mem)/start_mem)*100}%")
        
    def missing_value_analysis(self) ->pd.DataFrame:
        
        if self.df is None: return
        
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        report = pd.DataFrame({'Missing Values': missing, 'Percentage': missing_pct})
        
        logger.info("Missing Value Analysis Completed.")
        return report[report['Percentage'] > 0].sort_values(by='Percentage', ascending=False)
    
    def plot_rating_distribution(self) ->None:
        
        if self.df is None or 'overall' not in  self.df.columns: return
        
        plt.figure(figsize=(8,5))
        ax = sns.countplot(x='overall',data=self.df,palette='viridis')
        plt.title('Distribution of Product Ratings', fontsize=14)
        plt.xlabel('Rating', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        
         # Add exact counts on top of bars
        for p in ax.patches:
            ax.annotate(f'{p.get_height():,}', (p.get_x() + 0.4, p.get_height()), 
                        ha='center', va='bottom', fontsize=10)
        
        plt.show()
        
    def user_behaviour_analysis(self)->None:
        if self.df is None: return
        
        
              