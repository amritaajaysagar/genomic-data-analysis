import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def perform_eda(encoded_filepath):
    # Load encoded data
    train_data = pd.read_csv(encoded_filepath)
    
    # Set plot style
    sns.set(style="whitegrid")
    
    # Continuous variable distribution plots
    sns.displot(x='Patient_Age', data=train_data, kde=True)
    plt.title('Distribution of Patient Age')
    plt.show()
    
    sns.displot(x='Mothers_age', data=train_data, kde=True)
    plt.title('Distribution of Mothers Age')
    plt.show()
    
    sns.displot(x='Fathers_age', data=train_data, kde=True)
    plt.title('Distribution of Fathers Age')
    plt.show()
    
    # Genetic Disorder distribution
    plt.figure(figsize=(12, 4))
    sns.countplot(x='Genetic_Disorder', data=train_data, palette='pastel')
    plt.title('Genetic Disorder Distribution')
    plt.xlabel('Genetic Disorder')
    plt.ylabel('Count')
    plt.show()
    
    # Pie chart for Genetic_Disorder
    labels1 = [
        'Mitochondrial genetic inheritance disorders', 
        'Multifactorial genetic inheritance disorders', 
        'Single-gene inheritance diseases'
    ]
    explode1 = (0, 0, 0)
    
    plt.figure(figsize=(8, 8))
    plt.pie(train_data['Genetic_Disorder'].value_counts(), explode=explode1, labels=labels1, 
            autopct='%1.1f%%', shadow=True, startangle=90)
    plt.title('Genetic Disorder Proportions')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()
    
    # Disorder Subclass distribution
    plt.figure(figsize=(18, 6))
    sns.countplot(x='Disorder_Subclass', data=train_data, palette='Set2')
    plt.title('Disorder Subclass Distribution')
    plt.xlabel('Disorder Subclass')
    plt.ylabel('Count')
    plt.show()
    
    # Pie chart for Disorder Subclass
    labels2 = [
        'Leigh syndrome', 'Mitochondrial myopathy', 'Cystic fibrosis', 'Tay-Sachs', 
        'Diabetes', 'Hemochromatosis', "Leber's hereditary optic neuropathy", 
        "Alzheimer's", 'Cancer'
    ]
    explode2 = (0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    plt.figure(figsize=(8, 8))
    plt.pie(train_data['Disorder_Subclass'].value_counts(), explode=explode2, labels=labels2, 
            autopct='%1.1f%%', shadow=True, startangle=30)
    plt.title('Disorder Subclass Proportions')
    plt.axis('equal')
    plt.show()
    
    # Count plots for categorical features with Genetic Disorder hue
    cols = [
        'Genes_Mothers_Side', 'Inherited_Father', 'Maternal_gene', 'Paternal_gene', 
        'Gender', 'Birth_asphyxia', 'Autopsy_Birth_Defect', 
        'Folic_Acid', 'Maternal_Illness',
        'Radiation_Exposure', 'Substance_Abuse', 'Assisted_Conception', 'Birth_Defects'
    ]
    
    fig, ax = plt.subplots(len(cols), 1, figsize=(15, 45), constrained_layout=True)
    
    for i, var in enumerate(cols): 
        sns.countplot(data=train_data, x=var, hue='Genetic_Disorder', ax=ax[i], linewidth=1.5)
        ax[i].set_ylabel(var)
        ax[i].set_xlabel('') 
        ax[i].legend_.remove()
    
    plt.suptitle('Categorical Features Distribution by Genetic Disorder', fontsize=16)
    plt.show()

# Example usage
if __name__ == "__main__":
    perform_eda('encoded_data.csv')
