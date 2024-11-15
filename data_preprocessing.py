import pandas as pd

def data_preprocessing(filepath):
    # Load the dataset
    train_data = pd.read_csv(filepath)

    # Remove quotes and replace spaces with underscores
    train_data.columns = [col.strip().replace("'", '').replace(' ', '_') for col in train_data.columns]
    
    # Rename specific columns
    train_data = train_data.rename(columns={
        'Genes_in_mothers_side': 'Genes_Mothers_Side',
        'Inherited_from_father': 'Inherited_Father',
        'Blood_cell_count_(mcL)': 'Blood_Cell_mcL',
        'Respiratory_Rate_(breaths/min)': 'Respiratory_Rate_breaths_min',
        'Heart_Rate_(rates/min': 'Heart_Rates_Min',
        'Follow-up': 'Follow_up',
        'Autopsy_shows_birth_defect_(if_applicable)': 'Autopsy_Birth_Defect',
        'Folic_acid_details_(peri-conceptional)': 'Folic_Acid',
        'H/O_serious_maternal_illness': 'Maternal_Illness',
        'H/O_radiation_exposure_(x-ray)': 'Radiation_Exposure',
        'H/O_substance_abuse': 'Substance_Abuse',
        'Assisted_conception_IVF/ART': 'Assisted_Conception',
        'History_of_anomalies_in_previous_pregnancies': 'History_Previous_Pregnancies',
        'No._of_previous_abortion': 'Previous_Abortion',
        'Birth_defects': 'Birth_Defects',
        'White_Blood_cell_count_(thousand_per_microliter)': 'White_Blood_Cell',
    })

    # Drop rows where 'Genetic_Disorder' is NaN
    train_data.dropna(subset=['Genetic_Disorder'], inplace=True)
    train_data.reset_index(drop=True, inplace=True)

    # Drop unnecessary columns based on analysis
    drop_columns = [
        'Patient_Id', 'Patient_First_Name', 'Family_Name', 'Fathers_name', 
        'Institute_Name', 'Place_of_birth', 'Location_of_Institute', 
        'Test_1', 'Test_2', 'Test_3', 'Test_4', 'Test_5', 
        'Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4', 'Symptom_5'
    ]
    train_data.drop(columns=drop_columns, inplace=True)

    # Impute missing values for continuous features with median
    train_data['Patient_Age'].fillna(train_data['Patient_Age'].median(), inplace=True)
    train_data['Blood_Cell_mcL'].fillna(train_data['Blood_Cell_mcL'].median(), inplace=True)
    train_data['White_Blood_Cell'].fillna(train_data['White_Blood_Cell'].median(), inplace=True)

    # Impute missing values for categorical/ordinal features with mode
    columns_to_fill_mode = [
        'Genes_Mothers_Side', 'Inherited_Father', 'Maternal_gene', 'Paternal_gene', 'Mothers_age',
        'Fathers_age', 'Status', 'Respiratory_Rate_breaths_min', 'Heart_Rates_Min', 
        'Parental_consent', 'Follow_up', 'Gender', 'Birth_asphyxia', 'Autopsy_Birth_Defect',  
        'Folic_Acid', 'Maternal_Illness', 'Radiation_Exposure', 'Substance_Abuse', 
        'Assisted_Conception', 'History_Previous_Pregnancies', 'Previous_Abortion', 
        'Birth_Defects', 'Blood_test_result', 'Disorder_Subclass'
    ]
    for col in columns_to_fill_mode:
        train_data[col].fillna(train_data[col].mode()[0], inplace=True)

    # Save the preprocessed data
    train_data.to_csv('preprocessed_data.csv', index=False)
    print("Preprocessing completed. Saved to 'preprocessed_data.csv'.")

# Example usage
if __name__ == "__main__":
    data_preprocessing('train.csv')
