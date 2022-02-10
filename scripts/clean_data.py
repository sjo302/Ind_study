#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data cleaning function to be imported and used for all modeling. Allows for reproducibility
@author: saraokun, kpant, prestonlharry
"""

# ===============
# LIBRARIES
# ============
import json
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split


# ===============
# HELPER FUNCTIONS
# ============

def set_nulls(data):
    """
    Replace -99s (used in the raw data to designate null) of all forms with NaN values
    @param data: DataFrame of values
    @return: DF with NaN instead of -99
    """
    # Run three separate replaces, each for a different data type.
    data.replace(to_replace=-99, value=np.nan, inplace=True)
    data.replace(to_replace=-99.0, value=np.nan, inplace=True)
    data.replace(to_replace='-99', value=np.nan, inplace=True)
    return data


def map_cpt(data, column, replace, name):
    """
    Manually map principle CPT codes to predefined categories.
    @param data: DataFrame of values
    @param column: Column to apply changes to
    @param replace: Values to replace
    @param name: New value to replace with
    @return: DataFrame with replaced values
    """
    # Iterate through values to replace, replacing with name provided.
    for r in replace:
        idx = np.where(data[column] == r)[0]
        data[column].loc[idx] = name
    return data


def ensure_before_readmission(df, day_col, binary_col, cols_to_drop=[]):
    """
    Function used to ensure any event occurs before readmission, preventing leakage.
    @param df: DataFrame of values
    @param day_col: Column containing days to event occurrence
    @param binary_col: Column containing binary value (whether event occurred)
    @param cols_to_drop: Additional related columns not needed for analysis (optional)
    @return: DataFrame with all events occurring before readmission
    """
    days_to_readmission = df['READMPODAYS1'].replace(-99, 999)
    df.loc[(days_to_readmission - df[day_col]) <= 0, binary_col] = 0
    
    # Create list of columns to drop, combining provided list with now unneeded date column
    dropcols = cols_to_drop + [day_col]
    df.drop(columns=dropcols, inplace=True)
    return df


def replace_col_values(df, column, replace_dict, nan_vals=[], new_name=None):
    """
    Replace specified values in column with new values provided, set NaN and rename column where necessary
    @param df: DataFrame containing data
    @param column: Column to modify
    @param replace_dict: Dictionary old, new pairs for replacement
    @param nan_vals: Values to replace with NaN (optional)
    @param new_name: New column name for interpretability (optional)
    @return: Modified DataFrame
    """
    # Replace all values in column using dictionary provided
    df[column].replace(replace_dict, inplace=True)
    # Replace specific values with NaN
    for val in nan_vals:
        df[column].replace(val, np.nan, inplace=True)
    # Rename column if appropriate
    if new_name is not None:
        df.rename(columns={column: new_name}, inplace=True)
    return df


def le_ohe(data, features):
    """
    Label encoding and one-hot encoding for categorical variables.
    @param data: DataFrame of data to modify
    @param features: Features to encode
    @return: Encoded DataFrame
    """
    # Use SKLearn's LabelEncoder to create encodings for data. Store in dictionary
    le = LabelEncoder()
    encodings = {}
    for c in features:
        data[c] = le.fit_transform(data[c].astype(str))
        encodings[c] = dict(zip(le.classes_, le.transform(le.classes_)))

    # Write encodings to txt file for record-keeping + result interpretability
    with open('encodings.txt', 'w') as mappings:
        for key, value in encodings.items():
            mappings.write('%s:%s\n' % (key, value))

    # Use SKLearn's OneHotEncoder to encode categorical variables. Store in new DataFrame
    ohe = OneHotEncoder(drop='first')
    ohe.fit(data[features])
    ohe_labels = ohe.transform(data[features]).toarray()
    df_cat = pd.DataFrame(ohe_labels, columns=ohe.get_feature_names(features))

    # Drop old features, concatenate data with new encoded features
    data.drop(features, axis=1, inplace=True)
    clean_data = pd.concat([data, df_cat], axis=1)
    return clean_data


# ===============
# MAIN FUNCTION
# ============

def load_and_clean_data():
    """
    Function which loads in data and applies a series of cleaning operations.
    @return: Cleaned dataframe
    """
    # Read in data and cleaning JSON
    f = open('../data/data_cleaning.json', )
    clean_dict = json.load(f)
    df = pd.read_csv("../data/monet_output.csv")
    df.drop(['Unnamed: 0'], axis=1, inplace=True)

    # Deal with null values: replace -99s with NaN and fill NaN for certain columns where imputation is possible
    df = set_nulls(df)
    df['COL_APPROACH'].fillna("other", inplace=True)
    df['REOPERATION2'].fillna(0, inplace=True)

    # Categorize approach using predefined categories
    options = [clean_dict["approach"]['MIS'], clean_dict["approach"]['open'], clean_dict["approach"]['other']]
    names = ['MIS', 'open', 'other']
    for i in range(len(options)):
        df = map_cpt(df, 'COL_APPROACH', options[i], names[i])
    nulls = np.where(df.COL_APPROACH == 'Unknown')[0]
    df.COL_APPROACH.loc[nulls] = np.nan  # Set unknown values to NaN

    # Create new BMI column using height and weight
    df["bmi"] = (df.WEIGHT / 2.205) / ((df.HEIGHT / 39.37) ** 2)

    # Process target variable
    unplanned = [c for c in df if "UNPLANNEDREADMISSION" in c]
    df['target'] = [1 if x > 0 else 0 for x in df[unplanned].sum(axis=1)]

    # Process other procedures and concurrent procedures (subject to change)
    # Count number of other/concurrent procs
    othercpt = [c for c in df if "OTHERCPT" in c]
    df['num_other_procs'] = df[othercpt].count(axis=1)
    concurrcpt = [c for c in df if "CONCURR" in c]
    df['num_concurr_procs'] = df[concurrcpt].count(axis=1)

    # Binarize relevant groups within other/concurrent procs
    # Get procedure mappings and clean
    proc = pd.concat([df.filter(like="OTHERPROC"), df.filter(like = "CONCURR")], axis=1).copy()

    proc_maps = pd.read_csv('../data/procedure_maps.csv')
    proc_maps = proc_maps[proc_maps.Category.isna()==False]
    proc_maps["Procedure"] = proc_maps.Procedure.str.upper()
    proc_maps.drop(columns="prop_patients", inplace=True)
    proc_maps.rename(columns={"CPT Code":"cpt_code", "Category":"category", "Procedure":"procedure"}, inplace=True)
    proc_maps["category"] = np.where(proc_maps.category == "UROGENITAL, OBSTETRY",
         "UROGENITAL", proc_maps.category)
    proc_maps["category"] = np.where(proc_maps.category == "ENDOCRINE, NERVOUS, EYE, OCULAR ADNEXA,AUDITORY",
             "ENDOCRINE", proc_maps.category)
    proc_maps["category"] = np.where(proc_maps.category == "RESP, CV, HEMIC,  LYMPH",
             "RESP", proc_maps.category)
    proc_maps["category"] = np.where(proc_maps.category == "MEDICINE EVALUATION AND MANAGEMENT",
             "MED_EVAL", proc_maps.category)

    # Binarize procedure groups
    proc_cols = list(proc.columns.values)
    cat_names = []
    code_names = []

    for i in range(len(proc_cols)):
        proc = pd.merge(proc, proc_maps, how="left", left_on=proc_cols[i], right_on="procedure")
        cat_names.append("cat" + str(i+1))
        proc[cat_names[i]] = proc.category
        code_names.append("code" + str(i+1))
        proc[code_names[i]] = proc.cpt_code
        proc.drop(columns=["cpt_code", "category", "procedure"], inplace=True)

    proc["cat_list"] = [set([x for x in l if pd.isnull(x)==False]) for l in proc[cat_names].values.tolist()]
    proc["code_list"] = [set([x for x in l if pd.isnull(x)==False]) for l in proc[code_names].values.tolist()]

    proc.drop(columns=cat_names + code_names, inplace=True)

    proc_groups = list(proc_maps.category.unique())

    for cat in proc_maps.category.unique():
        temp_list = []
        for i in range(len(proc)):
            if cat in proc.cat_list[i]:
                temp_list.append(1)
            else:
                temp_list.append(0)
        proc[cat] = temp_list

    dig_groups = {}
    dig_groups["ORO_ESOPH"] = {"bot":42955, "top":43499}
    dig_groups["STOMACH"] = {"bot":43500, "top":43999}
    dig_groups["SM_INT"] = {"bot":44000, "top":44799}
    dig_groups["MECKEL"] = {"bot":44800, "top":44899}
    dig_groups["PROCTOLOGY"] = {"bot":44900, "top":46999}
    dig_groups["HEP_PAN_BIL"] = {"bot":47000, "top":48999}
    dig_groups["PERITONEUM"] = {"bot":48999, "top":49999}

    for grp in dig_groups.keys():
        temp_list = []
        for i in range(len(proc)):
            temp_val = 0
            for codes in proc.code_list[i]:
                if (codes >= dig_groups[grp]["bot"]) & (codes <= dig_groups[grp]["top"]):
                    temp_val+=1
            if temp_val > 0:
                temp_list.append(1)
            else:
                temp_list.append(0)

        proc[grp] = temp_list

    # Append procedure groups to full data
    final_procs = proc[list(proc_maps.category.unique()) + list(dig_groups.keys())]
    df = pd.concat([df, final_procs], axis=1)


    # Binarize relevant groups within PODIAG10
    podiag_maps = pd.read_csv('../data/podiag_maps.csv', usecols=["Prop", "Parent Code"])
    podiag_maps.rename(columns={"Prop":"prop", "Parent Code":"PODIAG10"}, inplace=True)

    podiag_maps = podiag_maps[podiag_maps.prop >= 0.001]

    podiag = df[["PODIAG10", "PODIAGTX10"]].copy()
    podiag.PODIAG10.replace('\.[0-9]+', '', regex=True, inplace=True)

    for po_grp in podiag_maps.PODIAG10:
        podiag[po_grp] = np.where(podiag.PODIAG10 == po_grp, 1, 0)

    # Append PODIAG groups to full data
    final_podiag = podiag.drop(columns=["PODIAG10", "PODIAGTX10"])
    df = pd.concat([df, final_podiag], axis=1)


    # Misc (diabetes and bleedis)
    df['insulin'] = df.DIABETES  # Duplicate diabetes column for one-hot encoding

    # Make sure that any relevant event occurs before readmission
    ensure_dict = clean_dict['ensure_before_readmission']
    for binary_col in ensure_dict.keys():
        df = ensure_before_readmission(df,
                                       ensure_dict[binary_col]['day_col'],
                                       binary_col,
                                       ensure_dict[binary_col]['cols_to_drop'])

    # Drop unneeded columns (DO THIS AFTER CREATING NEW COLUMNS AS DROP LIST MAY INCLUDE COLUMNS NEEDED)
    cols_to_drop = set(clean_dict['cols_to_drop'])
    cols_to_drop = cols_to_drop.intersection(set(df.columns))
    df.drop(columns=cols_to_drop, inplace=True)

    # Replace and rename columns for simplicity/interpretability/ML processing
    replace_dict = clean_dict['replace_col_vals']
    for col in replace_dict.keys():
        df = replace_col_values(df, col, **replace_dict[col])

    # Drop NaN values for age and sex and reset index before OHE before one-hot encoding
    df.dropna(subset=['AGE', 'female'], inplace=True)
    df.reset_index(inplace=True, drop=True)
    
    # Remove rows with null values in specific columns
    dropna_cols = set(clean_dict['dropna_cols'])
    dropna_cols = dropna_cols.intersection(set(df.columns))
    df.dropna(axis=0, subset=dropna_cols, inplace=True)
    
    # Apply one hot encoding to categorical variables
    cat_feat = [col for col in df.columns if (df[col].dtype == 'O') or (df[col].isnull().sum() != 0)]
    
    #OHE
    df = le_ohe(df, cat_feat)

    X = df.drop('target', axis=1)
    y = df['target']

    # 70 20 10 split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=42)

    return X_train, X_test, y_train, y_test


# ===============
# MAIN EXECUTION
# ============

if __name__ == "__main__":
    # If executing from this script directly, run and print shape of cleaned DataFrame for debugging
    X_train, X_test, y_train, y_test = load_and_clean_data()
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
