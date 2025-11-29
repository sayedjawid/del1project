import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA



class HealthAnalyzer:
    '''
        Klass för att analyzera health_study_dataset.csv, inkluderar statistik, grafer, enkel linjär regression och PCA.
    '''

    def __init__(self, df: pd.DataFrame):
        """
            Initierar klassen med pandas dataframe.
            dataframen inehåller kolumner ' age, weight, blood_pressure, sex, osv '
        """
        self.df = df.copy()

    def summary_status(self):
        """Returnerar glundlägande statistisk som 'medelvärde, std, min och max värde'"""
        return self.df.describe()
    

    def plot_age_vs_bp(self):
        """Scatter plot, ålder vs blodtryck"""
        plt.figure(figsize=(8,6))    
        plt.scatter(self.df['age'], self.df['systolic_bp'], alpha=0.7)
        plt.title('Ålder vs Blodtryck')
        plt.xlabel('Ålder')
        plt.ylabel('Blodtryck')
        plt.grid(True)
        plt.show()


    def plot_histogram(self, column:str, bins:int=10):
        """Histogram för en kolumn"""
        plt.figure(figsize=(8,6))
        plt.hist(self.df[column], bins=bins, color='blue', edgecolor='black')
        plt.title(f'Förändring av colimn {column}')
        plt.ylabel('Antal')
        plt.show()


    def group_mean(self, group_col:str, value_col:str):
        """Beräknar medelvärdet av value_col per kategori i group_col"""
        return self.df.groupby(group_col)[value_col].mean()
    

    def liner_regression(self, feature:str, target:str):
        """
            En enkel linjär regression, target och feature
            returnerar tränad modell, coef och intercept
        """
        x = self.df[[feature]].values.reshape(-1, 1)
        y = self.df[target].values
        model = LinearRegression().fit(x,y)
        coef = model.coef_[0]
        intercept = model.intercept_
        return model, coef, intercept
    

    def plot_regressionline(self, feature:str, target:str):
        """Scatterplot med Regressionlinje"""
        model, coef, intercept = self.liner_regression(feature, target)
        plt.figure(figsize=(8, 6))
        plt.scatter(self.df[feature], self.df[target], alpha=0.7, lable='Data')
        x_vals = np.array([self.df[feature].min(), self.df[feature].max()])
        y_vals = coef * x_vals, + intercept
        plt.plot(x_vals, y_vals, color='red', lable='Regression')
        plt.x_lable(feature)
        plt.ylabel(target)
        plt.title(f'{target} ~ {feature}')
        plt.legend()
        plt.grid(True)
        plt.show()


    def pca_analysis(self, n_components=2):
        """PCA_analys på numeriska kolumner"""
        numeric_df = self.df.select_dtypes(include=np.number)
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(numeric_df)
        explained_var = pca.explained_variance_ratio_


        plt.figure(figsize=(8,6))
        plt.scatter(components[:,0], components[:,1], alpha=0.7)
        plt.title('PCA: Första två komponenterna')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.grid(True)
        plt.show()
        
        return components, explained_var
    
