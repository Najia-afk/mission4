import pandas as pd
import numpy as np
import plotly.express as px

def check_property_usage_coherence(df):
    """Check consistency between different property measurements and return issues."""
    results = []
    
    # Check if total area corresponds to sum of areas (building + parking)
    if all(col in df.columns for col in ['PropertyGFATotal', 'PropertyGFAParking', 'PropertyGFABuilding(s)']):
        df_numeric = df.copy()
        df_numeric['PropertyGFATotal'] = pd.to_numeric(df_numeric['PropertyGFATotal'], errors='coerce')
        df_numeric['PropertyGFAParking'] = pd.to_numeric(df_numeric['PropertyGFAParking'], errors='coerce')
        df_numeric['PropertyGFABuilding(s)'] = pd.to_numeric(df_numeric['PropertyGFABuilding(s)'], errors='coerce')
        
        # Calculate difference
        df_numeric['GFA_Difference'] = df_numeric['PropertyGFATotal'] - (df_numeric['PropertyGFAParking'] + df_numeric['PropertyGFABuilding(s)'])
        
        # Identify significant inconsistencies (more than 1% difference)
        gfa_issues = df_numeric[abs(df_numeric['GFA_Difference']) > df_numeric['PropertyGFATotal'] * 0.01].shape[0]
        results.append(f"Area inconsistencies: {gfa_issues} buildings ({gfa_issues/len(df)*100:.2f}%)")
    
    # Check consistency between main usage types and the complete list
    if all(col in df.columns for col in ['ListOfAllPropertyUseTypes', 'LargestPropertyUseType']):
        # Fixed approach: Check row by row if LargestPropertyUseType is in ListOfAllPropertyUseTypes
        missing_count = 0
        for _, row in df.iterrows():
            if pd.notna(row['ListOfAllPropertyUseTypes']) and pd.notna(row['LargestPropertyUseType']):
                if row['LargestPropertyUseType'] not in str(row['ListOfAllPropertyUseTypes']).split(', '):
                    missing_count += 1
        results.append(f"Primary use missing in complete list: {missing_count} buildings ({missing_count/len(df)*100:.2f}%)")
    
    return results

def handle_parking_in_property_types(df):
    """Remove parking from property use types and promote lower usage types."""
    temp_df = df.copy()
    
    # Define the columns to check for parking
    usage_type_cols = ['LargestPropertyUseType', 'SecondLargestPropertyUseType', 'ThirdLargestPropertyUseType']
    gfa_cols = ['LargestPropertyUseTypeGFA', 'SecondLargestPropertyUseTypeGFA', 'ThirdLargestPropertyUseTypeGFA']
    
    # Ensure all GFA columns are numeric
    for col in gfa_cols:
        temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce').fillna(0)
    
    # Track how many buildings had parking handled
    buildings_modified = 0
    
    # Process each row
    for idx, row in temp_df.iterrows():
        modified = False
        
        # Check if any usage type is 'Parking'
        for i, (type_col, gfa_col) in enumerate(zip(usage_type_cols, gfa_cols)):
            if pd.notna(row[type_col]) and row[type_col] == 'Parking':
                # Remove this parking usage
                temp_df.at[idx, type_col] = 'None'
                temp_df.at[idx, gfa_col] = 0
                modified = True
                
                # Shift up all subsequent usage types
                for j in range(i+1, len(usage_type_cols)):
                    if pd.notna(row[usage_type_cols[j]]) and row[usage_type_cols[j]] != 'None':
                        temp_df.at[idx, usage_type_cols[j-1]] = row[usage_type_cols[j]]
                        temp_df.at[idx, gfa_cols[j-1]] = row[gfa_cols[j]]
                        temp_df.at[idx, usage_type_cols[j]] = 'None'
                        temp_df.at[idx, gfa_cols[j]] = 0
        
        if modified:
            buildings_modified += 1
    
    print(f"Parking usage removed from {buildings_modified} buildings ({buildings_modified/len(temp_df)*100:.2f}%)")
    return temp_df

def calculate_surface_and_ratios(df):
    """Calculate surface areas and ratios for building usage."""
    temp_df = df.copy()
    
    # First, handle parking in property types
    temp_df = handle_parking_in_property_types(temp_df)
    
    # Convert necessary columns to numeric
    for col in ['LargestPropertyUseTypeGFA', 'SecondLargestPropertyUseTypeGFA', 'ThirdLargestPropertyUseTypeGFA']:
        if col in temp_df.columns:
            temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce').fillna(0)
    
    # Calculate total area of main uses
    temp_df['BuildingTotalSurface'] = temp_df['LargestPropertyUseTypeGFA'] + temp_df['SecondLargestPropertyUseTypeGFA'] + temp_df['ThirdLargestPropertyUseTypeGFA']
    
    # Calculate area ratios
    for prefix, col in zip(['Largest', 'Second', 'Third'], 
                         ['LargestPropertyUseTypeGFA', 'SecondLargestPropertyUseTypeGFA', 'ThirdLargestPropertyUseTypeGFA']):
        if col in temp_df.columns:
            temp_df[f'{prefix}SurfaceRatio'] = temp_df[col] / temp_df['BuildingTotalSurface']
            # Replace inf/nan values with 0
            temp_df[f'{prefix}SurfaceRatio'] = temp_df[f'{prefix}SurfaceRatio'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return temp_df

def clean_and_transform_features(df):
    """Perform comprehensive cleaning and feature engineering."""
    temp_df = df.copy()
    
    # Handle missing values for areas
    surface_cols = ['LargestPropertyUseTypeGFA', 'SecondLargestPropertyUseTypeGFA', 'ThirdLargestPropertyUseTypeGFA', 'PropertyGFAParking']
    for col in surface_cols:
        if col in temp_df.columns:
            temp_df[col] = temp_df[col].fillna(0).astype(float)
    
    # Handle missing values for usage types
    usage_cols = ['LargestPropertyUseType', 'SecondLargestPropertyUseType', 'ThirdLargestPropertyUseType']
    for col in usage_cols:
        if col in temp_df.columns:
            temp_df[col] = temp_df[col].fillna('None')
    
    # Handle energy sources
    energy_cols = ['SteamUse(kBtu)', 'Electricity(kBtu)', 'NaturalGas(kBtu)']
    for col in energy_cols:
        if col in temp_df.columns:
            temp_df[col] = temp_df[col].fillna(0).astype(float)
    
    # Create total energy variable
    if all(col in temp_df.columns for col in energy_cols):
        temp_df['TotalEnergy(kBtu)'] = temp_df['SteamUse(kBtu)'] + temp_df['Electricity(kBtu)'] + temp_df['NaturalGas(kBtu)']
    
    # Create percentages by energy source
    if 'TotalEnergy(kBtu)' in temp_df.columns:
        for source, col in zip(['SteamUse', 'Electricity', 'NaturalGas'], energy_cols):
            temp_df[source] = (temp_df[col] / temp_df['TotalEnergy(kBtu)']) * 100
            # Replace NaN with 0 (case where TotalEnergy is 0)
            temp_df[source] = temp_df[source].fillna(0)
        
        # Remove original energy columns
        temp_df = temp_df.drop(columns=energy_cols)

    if 'YearBuilt' in temp_df.columns:
        # Calculate building age (as of 2016 when data was collected)
        temp_df['BuildingAge'] = 2016 - temp_df['YearBuilt']
        # Handle any invalid ages (negative or extremely large)
        temp_df['BuildingAge'] = temp_df['BuildingAge'].clip(0, 200)
    
    # Add usage intensity metrics (normalize energy by building characteristics)
    if ('TotalEnergy(kBtu)' in temp_df.columns) and False:
        # Energy per square foot (energy intensity)
        if 'PropertyGFATotal' in temp_df.columns:
            temp_df['EnergyPerSqFt'] = temp_df['TotalEnergy(kBtu)'] / pd.to_numeric(temp_df['PropertyGFATotal'], errors='coerce')
            temp_df['EnergyPerSqFt'] = temp_df['EnergyPerSqFt'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Energy per floor 
        if 'NumberofFloors' in temp_df.columns:
            temp_df['EnergyPerFloor'] = temp_df['TotalEnergy(kBtu)'] / pd.to_numeric(temp_df['NumberofFloors'], errors='coerce')
            temp_df['EnergyPerFloor'] = temp_df['EnergyPerFloor'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Energy per building (for multi-building properties)
        if 'NumberofBuildings' in temp_df.columns:
            temp_df['EnergyPerBuilding'] = temp_df['TotalEnergy(kBtu)'] / pd.to_numeric(temp_df['NumberofBuildings'], errors='coerce')
            temp_df['EnergyPerBuilding'] = temp_df['EnergyPerBuilding'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
    # Convert important numeric columns
    numeric_cols = ['TotalEnergy(kBtu)', 'TotalGHGEmissions'
                   'BuildingAge', 'CouncilDistrictCode', 'NumberofBuildings', 
                   'NumberofFloors', 'ENERGYSTARScore']
    
    for col in numeric_cols:
        if col in temp_df.columns:
            temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')
    
    # Remove unnecessary columns
    cols_to_drop = ['BuildingType', 'PrimaryPropertyType', 'YearBuilt', 'PropertyGFATotal', 'PropertyGFAParking', 'PropertyGFABuilding(s)',
                    'ListOfAllPropertyUseTypes', 'LargestPropertyUseTypeGFA','SecondLargestPropertyUseTypeGFA', 'ThirdLargestPropertyUseTypeGFA',
                    'SiteEUI(kBtu/sf)', 'SiteEUIWN(kBtu/sf)', 'SourceEUI(kBtu/sf)', 'SourceEUIWN(kBtu/sf)', 'SiteEnergyUse(kBtu)', 'SiteEnergyUseWN(kBtu)',
                    'SteamUse(kBtu)', 'Electricity(kWh)', 'Electricity(kBtu)', 'NaturalGas(therms)', 'NaturalGas(kBtu)','GHGEmissionsIntensity',
                    'Latitude','Longitude','GHGEmissionsIntensity_normalized', 'SiteEnergyUse(kBtu)_normalized', 'TotalGHGEmissions_normalized', 'ENERGYSTARScore_normalized']

    cols_to_drop = [col for col in cols_to_drop if col in temp_df.columns]
    temp_df = temp_df.drop(columns=cols_to_drop)
    
    return temp_df

def process_building_data(df_clean, print_results=True):
    """Main function to run the entire data processing pipeline."""
    # Check data coherence
    coherence_results = check_property_usage_coherence(df_clean)
    if print_results:
        print("Coherence check results:")
        for result in coherence_results:
            print(f"- {result}")
    
    # Calculate surface ratios
    df_with_ratios = calculate_surface_and_ratios(df_clean)
    
    # Clean and transform features
    df_transformed = clean_and_transform_features(df_with_ratios)
    
    return df_transformed, coherence_results