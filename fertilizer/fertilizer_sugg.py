import pandas as pd
from .fertilizer_dict import fertilizer_dic  # Import the fertilizer dictionary

def suggest_fertilizer(crop_name, user_N, user_P, user_K):
    # df = pd.read_csv('C:\Users\arfat\OneDrive\Pictures\Desktop\AgroSense\datasets\fertilizer_data_processed.csv')
    df = pd.read_csv("./datasets/fertilizer_data_processed.csv")  # Load the fertilizer data
    crop_row = df[df['Crop'].str.lower() == crop_name.lower()]  # Match crop name (case-insensitive)
    if crop_row.empty:
        return f"Crop '{crop_name}' not found in database."  # Handle crop not found

    ideal_N = crop_row['N'].values[0]  # Ideal nitrogen value
    ideal_P = crop_row['P'].values[0]  # Ideal phosphorous value
    ideal_K = crop_row['K'].values[0]  # Ideal potassium value

    diff_N = ideal_N - user_N  # Difference in nitrogen
    diff_P = ideal_P - user_P  # Difference in phosphorous
    diff_K = ideal_K - user_K  # Difference in potassium

    differences = {abs(diff_N): 'N', abs(diff_P): 'P', abs(diff_K): 'K'}  # Find max nutrient difference
    key_nutrient = differences[max(differences.keys())]  # Identify nutrient needing attention

    if key_nutrient == 'N':
        if diff_N > 0:
            key = 'Nlow'  # Your soil has LESS nitrogen than needed
        else:
            key = 'NHigh'  # Your soil has MORE nitrogen than needed
    elif key_nutrient == 'P':
        if diff_P > 0:
            key = 'Plow'  # Your soil has LESS phosphorus than needed
        else:
            key = 'PHigh'  # Your soil has MORE phosphorus than needed
    else:  # key_nutrient == 'K'
        if diff_K > 0:
            key = 'Klow'  # Your soil has LESS potassium than needed
        else:
            key = 'KHigh'  # Your soil has MORE potassium than needed

    return fertilizer_dic[key]  # Return fertilizer recommendation