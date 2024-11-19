import pandas as pd

def encode_data(input_filepath, output_filepath):
    # Load preprocessed data
    train_data = pd.read_csv(input_filepath)
    
    # Columns to encode
    columns_to_encode = [
        "Genes_Mothers_Side", "Inherited_Father", "Maternal_gene", "Paternal_gene", "Status",
        "Respiratory_Rate_breaths_min", "Heart_Rates_Min", "Parental_consent", "Follow_up",
        "Gender", "Birth_asphyxia", "Autopsy_Birth_Defect", "Folic_Acid", "Maternal_Illness",
        "Radiation_Exposure", "Substance_Abuse", "Assisted_Conception", 
        "History_Previous_Pregnancies", "Birth_Defects", "Blood_test_result", 
        "Genetic_Disorder", "Disorder_Subclass"
    ]
    
    # Dictionary to store mappings for each column (optional, for reference)
    encoding_mappings = {}
    
    # Encode each column using Label Encoding
    for col in columns_to_encode:
        train_data[col], uniques = pd.factorize(train_data[col])
        encoding_mappings[col] = {index: value for index, value in enumerate(uniques)}
    
    # Optionally, save the encoding mappings for future reference
    # import json
    # with open('encoding_mappings.json', 'w') as f:
    #     json.dump(encoding_mappings, f)
    
    # Save the encoded data
    train_data.to_csv(output_filepath, index=False)
    print(f"Encoding completed. Encoded data saved to '{output_filepath}'.")

    # Optionally, print encoding mappings
    for col, mapping in encoding_mappings.items():
        print(f"Encoding mapping for '{col}':")
        for encoded_value, original_value in mapping.items():
            print(f"  {encoded_value}: {original_value}")
        print("\n")  # Blank line between columns for readability

# Example usage
if __name__ == "__main__":
    encode_data('preprocessed_data.csv', 'encoded_data.csv')
