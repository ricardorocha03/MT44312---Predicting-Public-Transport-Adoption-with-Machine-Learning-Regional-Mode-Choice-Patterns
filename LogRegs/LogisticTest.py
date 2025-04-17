from itertools import product
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score, f1_score, confusion_matrix

def run_pipeline(Filltype, dataAugmentation, CatDealer, use_attitudes):
    # Test Size (usually 0.2)
    TestSize = 0.2

    # Set RandomState = None to have true randomness
    RandomState = 42

    # region Gather Data, Remove incomplete rows, and remove any NaN
    data = pd.read_csv('ModeChoiceOptima.txt', sep="\t", header=None)

    # Renaming correctly the columns of pandas data
    columnsID = data.columns
    featureNames = data.iloc[0, :]
    NameDictionary = dict(zip(columnsID, featureNames))
    data.rename(columns=NameDictionary, inplace=True)
    data.drop(index=data.index[0], axis=0, inplace=True)

    data.drop((data[data.Choice == '-1']).index, inplace=True)
    data.drop((data[data.Choice == '-2']).index, inplace=True)
    data.drop((data[data.Choice == '2']).index, inplace=True)

    # Replacing all other '-1' and '-2' with NaN for consistency
    data.replace('-1', np.nan, inplace=True)
    data.replace('-2', np.nan, inplace=True)
    data.drop((data[data.Choice == '2']).index, inplace=True)

    # Changing the dtype to numerical values, because it should help speed things up
    data = data.apply(pd.to_numeric, errors='raise')

    # Removing the attitude questions and unwanted columns
    DelColumns = ['ID', 'Weight', 'CostCar', 'CoderegionCAR']
    # (Deleting also CoderegionCAR as it's the same as 'Region')
    if not use_attitudes:
        DelColumns += [f'Envir{i:02d}' for i in range(1, 7)]
        DelColumns += [f'Mobil{i:02d}' for i in range(1, 28)]
        DelColumns += [f'ResidCh{i:02d}' for i in range(1, 8)]
        DelColumns += [f'LifSty{i:02d}' for i in range(1, 15)]
        # Also the attitude questions get deleted
        for col in DelColumns:
            data.drop(col, axis = 1, inplace = True)

    data.to_csv("DataUsedColumns.csv", index=False)


    # endregion

    # region Dummy Variables
    ReqDummy = ['DestAct', 'HouseType', 'Mothertongue', 'FamilSitu', 'OccupStat', 'SocioProfCat', 'Education',
                'TripPurpose', 'TypeCommune', 'ClassifCodeLine', 'ResidChild', 'Region', 'ModeToSchool']

    dummydata = pd.DataFrame()
    for feat in ReqDummy:
        if feat in data.columns:
            dummydata[feat] = data[feat]
        else:
            print(f"Column '{feat}' not found in data")

        # Deleting the original column in the main dataframe
        data.drop(feat, axis=1, inplace=True)

    def dummy_inator(X, na_indicator=True):
        DumbOutput = pd.DataFrame()
        columns = X.columns
        for feat in columns:
            dummies = pd.get_dummies(X[feat], prefix=feat, dummy_na=na_indicator)
            DumbOutput = pd.concat([DumbOutput, dummies], axis=1)
        return DumbOutput
    # endregion

    # region Split into 3 dataframes
    # Splitting everything in 3 different Dataframes 1) Choice 2) Cathegorical features (X_dummy) 3) Non-catheghorical (X)
    y = data['Choice']
    # print("y shape: ", y.shape)
    X = data.drop(columns=['Choice'])
    X_dummy = dummydata
    # endregion

    # region Scale Data
    scaler = StandardScaler()
    temp_scaled = scaler.fit_transform(X)

    # create a new DataFrame with the scaled data
    X_scaled = pd.DataFrame(temp_scaled, columns=X.columns)

    # Checking that the dimensions are right and amount of NaN entries
    y = y.reset_index(drop=True)
    X_scaled = X_scaled.reset_index(drop=True)
    X_dummy = X_dummy.reset_index(drop=True)
    # endregion

    # region Deal with Missing Data
    # Making the data split before imputation and/or augmentation
    def make_custom_split(X_scaled, X_dummy, y, valid_rows):

        # Create clean version
        X_scaled_clean = X_scaled[valid_rows]

        # Train-test split using indices
        X_indices = X_scaled_clean.index  # Save original indices
        train_idx, test_idx = train_test_split(X_indices, test_size=TestSize, random_state=RandomState)

        # the test datapoints are good and will be used
        X_test_scaled = X_scaled.loc[test_idx]
        X_test_dummy = X_dummy.loc[test_idx]
        y_test = y.loc[test_idx]

        # Update the original X_scaled and X_dummy to exclude the test rows
        X_scaled = X_scaled.drop(index=test_idx)
        X_dummy = X_dummy.drop(index=test_idx)
        y = y.drop(index=test_idx)

        return X_test_scaled, X_test_dummy, y_test, X_scaled, X_dummy, y

    # DEALING WITH MISSING DATA
    if CatDealer:
        # Identify rows in X_dummy that contain NaN values
        nan_mask = X_dummy.isna().any(axis=1)  # True for rows with any NaN
        # Drop those rows in both X_dummy, X_scaled and y
        X_dummy_cleaned = X_dummy[~nan_mask].reset_index(drop=True)
        X_scaled_cleaned = X_scaled[~nan_mask].reset_index(drop=True)
        y_cleaned = y[~nan_mask].reset_index(drop=True)
        X_WithDummies = dummy_inator(X_dummy_cleaned, False)

        # Identify valid rows (no NaNs in either dataset)
        valid_rows = (~X_scaled_cleaned.isna().any(axis=1)) & (~X_dummy_cleaned.isna().any(axis=1))
        X_test_scaled, X_test_dummy, y_test, X_scaled, X_dummy, y = make_custom_split(X_scaled_cleaned, X_WithDummies,
                                                                                      y_cleaned, valid_rows)

    else:
        X_dummy_cleaned = X_dummy
        X_scaled_cleaned = X_scaled
        y_cleaned = y
        X_WithDummies = dummy_inator(X_dummy_cleaned, True)

        # Identify valid rows (no NaNs in either dataset)
        valid_rows = (~X_scaled_cleaned.isna().any(axis=1)) & (~X_dummy_cleaned.isna().any(axis=1))
        X_test_scaled, X_test_dummy, y_test, X_scaled, X_dummy, y = make_custom_split(X_scaled_cleaned, X_WithDummies,
                                                                                      y_cleaned, valid_rows)

    # Joining the Categorical and non-Categorical features, the last column will be the choice one
    data = pd.concat([X_scaled, X_dummy, y], axis=1)
    data_test = pd.concat([X_test_scaled, X_test_dummy, y_test], axis=1)
    # At this stage, data_test is ready

    data.to_csv("DataPreFilled.csv", index=False)

    match Filltype:
        case 0:
            # drop all rows with NaN entries
            data = data.dropna()
        case 1:
            data = data.ffill()
        case 2:
            # SimpleImputer - most_frequent
            imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            data2 = pd.DataFrame(imp.fit_transform(data))
            data2.columns = data.columns
            data2.index = data.index
            data = data2
        case 3:
            # This method is quite expensive computationally!
            imputer = IterativeImputer(max_iter=15, random_state=RandomState, initial_strategy='mean')
            data2 = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
            data = data2
        case 4:
            # SimpleImputer - mean
            imp = SimpleImputer(missing_values=np.nan, strategy='mean')
            data2 = pd.DataFrame(imp.fit_transform(data))
            data2.columns = data.columns
            data2.index = data.index
            data = data2

    # extracting the last column (Choice)
    y = data.iloc[:, -1].values
    data.drop(data.columns[-1], axis=1, inplace=True)
    X_tot = data.values
    y_test = data_test.iloc[:, -1].values
    data_test.drop(data_test.columns[-1], axis=1, inplace=True)
    X_tot_test = data_test.values

    data.to_csv("DataFilled.csv", index=False)
    # endregion

    # region Augment and Split Data

    def augment_data(X, y):
        # Apply SMOTE to integer labels
        smote = SMOTE(sampling_strategy='auto', random_state=RandomState)
        X_train_resampled, y_train_resampled = smote.fit_resample(X, y)

        return X_train_resampled, y_train_resampled

    # Augment and split the data
    X_data = X_tot.astype(np.float32)
    y_data = y.astype(int)

    X_test = X_tot_test.astype(np.float32)
    y_num_test = y_test.astype(int)

    if dataAugmentation:
        y_test_LOG = y_num_test
        X_train, y_train_LOG = augment_data(X_data, y_data)
    else:
        y_train_LOG = y_data
        y_test_LOG = y_num_test
        X_train = X_data
    # endregion

    # region Actual Logistic Regression
    # -- Train logistic regression model
    logr = LogisticRegression(max_iter=500)
    logr.fit(X_train, y_train_LOG)

    # -- Predictions
    y_pred = logr.predict(X_test)

    print("\nClassification Report:\n",
          classification_report(y_test_LOG, y_pred, target_names=['PT 0', 'Private modes 1']))

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()
    # endregion

    return acc, prec, f1, cm


# region Test Different Configs

results = []
configs = list(product(range(5), [True, False], [0, 1], [True, False]))

for i, (fill, aug, cat, att) in enumerate(configs):
    print(f"\nRunning config {i + 1}/40 --> Filltype={fill}, Aug={aug}, CatDealer={cat}, Attitudes={att}")
    acc, prec, f1, cm = run_pipeline(fill, aug, cat, att)
    results.append({
        'Config_ID': i + 1,
        'Filltype': fill,
        'DataAug': aug,
        'CatDealer': cat,
        'AttitudesUsed': att,
        'Accuracy': acc,
        'Precision': prec,
        'F1_Score': f1,
        'Confusion_Matrix': cm
    })

results_df = pd.DataFrame(results)
results_df.to_csv("LR_Comparison_Results.csv", index=False)
print("\ All results saved to 'LR_Comparison_Results.csv'")


