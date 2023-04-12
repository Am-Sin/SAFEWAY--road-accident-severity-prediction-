# %%

# %% [markdown]
# ## IMPORTING LIBRARIES

# %%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import joblib
from PIL import Image
import PIL



# %%


# %% [markdown]
# ## IMPORTING THE DATASET 

# %%
df = pd.read_csv("RTA Dataset.csv")

# %% [markdown]
# ## EXPLORATORY DATA ANALYSIS

# %%
df.head()

# %%
df.tail()

# %%
df.info()

# %%
df.isnull().sum()

# %%
print(df['Accident_severity'].value_counts())
df['Accident_severity'].value_counts().plot(kind='bar')

# %%
print(df['Accident_severity'].value_counts())
df['Accident_severity'].value_counts().plot(kind='hist', color ='black')

# %%
df['Educational_level'].value_counts().plot(kind='bar')

# %%
df['Educational_level'].value_counts().plot(kind='pie')

# %%
plt.figure(figsize=(6,5))
sns.catplot(x='Educational_level', y='Accident_severity', data=df)
plt.xlabel("Educational level")
plt.xticks(rotation=60)
plt.show()

# %% [markdown]
# ## THE FOLLOWING CAN BE CONCLUDED FROM THE INITIAL DATA ANALYSIS 
# 

# %%
# -More the Number of casualties, higher the chances of fatal injuries at accident site
# -More the vehicles involved higher the chances of Serious injury
# -Light_conditions being darkness can cause higher serious injury
# -data is highly imbalanced
# -Features like area_accident_occured, Cause_of_accident, Day_of_week, type_of_junction seem to be imporatant features causing fatal injuries
# -Road_surface and road conditions do not affect fatal or serious accidents apparently

# %% [markdown]
# ## EFFECT OF ROAD SURFACE ON INJURIES 

# %%
print(df['Road_surface_type'].value_counts())

plt.figure(figsize=(6,5))
sns.countplot(x='Road_surface_type', hue='Accident_severity', data=df)
plt.xlabel('Rode surafce type')
plt.xticks(rotation=60)
plt.show

# %%
print(df['Road_surface_conditions'].value_counts())

plt.figure(figsize=(6,5))
sns.countplot(x='Road_surface_conditions', hue='Accident_severity', data=df)
plt.xlabel('Rode condition type')
plt.xticks(rotation=60)
plt.show

# %%
pivot_df = pd.pivot_table(data=df, 
               index='Road_surface_conditions', 
               columns='Accident_severity',
               aggfunc='count')

fatal_df = pivot_df['Road_surface_type']
fatal_df.fillna(0, inplace=True)
fatal_df['sum_of_injuries'] = fatal_df['Fatal injury'] + fatal_df['Serious Injury'] + fatal_df['Slight Injury']
fatal_df

# %%
fatal_df_dry = (fatal_df.loc['Dry']/fatal_df.loc['Dry','sum_of_injuries'])*100
fatal_df_dry

# %%
fatal_df_snow = (fatal_df.loc['Wet or damp']/fatal_df.loc['Wet or damp','sum_of_injuries'])*100
fatal_df_snow

# %%
df.groupby('Road_surface_conditions')['Accident_severity'].count()

# %% [markdown]
# ## CONVERTING "TIME" FEATURE INTO DATETIME FORMAT

# %%
df['Time'] = pd.to_datetime(df['Time'])

# %%
obj_cols = [col for col in df.columns if df[col].dtypes == 'object']
obj_cols2 = [col for col in obj_cols if col != 'Accident_severity']
obj_cols2

# %%
new_df = df.copy()
new_df['Hour_of_Day'] = new_df['Time'].dt.hour
n_df = new_df.drop('Time', axis=1)
n_df

# %% [markdown]
# ## DATA VISUALIZATION

# %%
def count_plot(col):
    n_df[col].value_counts()
    
    # plot the figure of count plot
    plt.figure(figsize=(5,5))
    sns.countplot(x=col, hue='Accident_severity', data=n_df)
    plt.xlabel(f'{col}')
    plt.xticks(rotation=60)
    plt.show
    
for col in obj_cols:
    count_plot(col)

# %%
plt.figure(figsize=(5,5))
sns.displot(x='Hour_of_Day', hue='Accident_severity', data=n_df)
plt.show()

# %% [markdown]
# ## FINAL INSIGHTS AND FURTHER STEPS
# 
# 

# %%
# -Hour_of_Day seems important to predict target
# -Lots of redundant features needs to be removed
# -Initially, Total 18 features are selected based on EDA and basic understanding of domain knowledge
# -Handle the missing values in these 20 features
# -Feature selection using scikit-libraries
# -Encoding categorical features
# -Handle imbalance dataset 
# -features standardization 
# -modelling
# -Evalution and Hyper-parameter tuning
# -selecting 17 features based on EDA insights and their informativeness to target feature

# %% [markdown]
# ## DATA PREPROCESSING

# %%
features = ['Day_of_week','Number_of_vehicles_involved','Number_of_casualties','Area_accident_occured',
           'Types_of_Junction','Age_band_of_driver','Sex_of_driver','Educational_level',
           'Vehicle_driver_relation','Type_of_vehicle','Driving_experience','Service_year_of_vehicle','Type_of_collision',
           'Sex_of_casualty','Age_band_of_casualty','Cause_of_accident','Hour_of_Day']
len(features)

# %%
featureset_df = n_df[features]
target = n_df['Accident_severity']

# %%
featureset_df.info()

# %% [markdown]
# ## MISSING VALUE TREATMENT 

# %%
feature_df = featureset_df.copy()

# %% [markdown]
# ### REPLACING THE MISSING VALUES WITH "Unknown"

# %%
feature_df['Service_year_of_vehicle'] = feature_df['Service_year_of_vehicle'].fillna('Unknown')
feature_df['Types_of_Junction'] = feature_df['Types_of_Junction'].fillna('Unknown')
feature_df['Area_accident_occured'] = feature_df['Area_accident_occured'].fillna('Unknown')
feature_df['Driving_experience'] = feature_df['Driving_experience'].fillna('unknown')
feature_df['Type_of_vehicle'] = feature_df['Type_of_vehicle'].fillna('Other')
feature_df['Vehicle_driver_relation'] = feature_df['Vehicle_driver_relation'].fillna('Unknown')
feature_df['Educational_level'] = feature_df['Educational_level'].fillna('Unknown')
feature_df['Type_of_collision'] = feature_df['Type_of_collision'].fillna('Unknown')

# %%
feature_df.info()

# %% [markdown]
# ## ONE HOT ENCODING 

# %%
X = feature_df[features]
y = target

# %%
encoded_df = pd.get_dummies(X, drop_first=True)
encoded_df.shape

# %% [markdown]
# ## LABEL ENCODING OF TARGET FEATURE 

# %%
lb = LabelEncoder()
lb.fit(y)
y_encoded = lb.transform(y)
print("Encoded labels:",lb.classes_)
y_en = pd.Series(y_encoded)

# %% [markdown]
# ### FEATURE SELECTION USING K BEST CHI2 METHOD

# %%
mi_calc = mutual_info_classif(encoded_df, y_en, random_state=42)

# %%
mi_df = pd.DataFrame({'Columns':encoded_df.columns, 'MI_score':mi_calc})
mi_df.sort_values(by='MI_score',ascending=False).head(15)

# %%
fs = SelectKBest(chi2, k=50)
X_new = fs.fit_transform(encoded_df, y_en)
X_new.shape
cols = fs.get_feature_names_out()

# %%
fs_df = pd.DataFrame(X_new, columns=cols)

# %% [markdown]
# ## PRINCIPAL COMPONENT ANALYSIS 

# %%
pca = PCA(n_components=3)
pca.fit(encoded_df)

X_pca = pca.transform(encoded_df)

components = pca.components_

pca_df = pd.DataFrame(X_pca, columns=["PC1","PC2","PC3"])
pca_df.var()

# %% [markdown]
# ### IMBALANCE DATA TREATMENT 

# %%
n_cat_index = np.array(range(3,50))

smote = SMOTENC(categorical_features=n_cat_index, random_state=42, n_jobs=True)
X_n, y_n = smote.fit_resample(fs_df,y_en)
X_n.shape, y_n.shape

# %%
y_n.value_counts()

# %% [markdown]
# ## MODELLING BASELINE AND HYPERPARAMTER TUNING OF RANDOM FOREST CLASSIFIER

# %%
# train and test split and building baseline model to predict target features
X_trn, X_tst, y_trn, y_tst = train_test_split(X_n, y_n, test_size=0.2, random_state=42)

# modelling using random forest baseline
rf = RandomForestClassifier(n_estimators=800, max_depth=20, random_state=42)
rf.fit(X_trn, y_trn)

# predicting on test data
predics = rf.predict(X_tst)

# %%
rf.score(X_trn, y_trn)

# %%
classif_re = classification_report(y_tst,predics)
print(classif_re)

# %%
conf_matrix = confusion_matrix(y_tst, predics)
conf_matrix

# %%
f1score = f1_score(y_tst,predics, average='weighted')
print(f1score)

# %% [markdown]
# ## MODELLING FOR DEPLOYMENT ON STREAMLIT CLOUD

# %%
cat_fea_df = feature_df.drop(['Hour_of_Day','Number_of_vehicles_involved','Number_of_casualties'], axis=1)

oencoder = OrdinalEncoder()
encoded_df2 = pd.DataFrame(oencoder.fit_transform(cat_fea_df))
encoded_df2.columns = cat_fea_df.columns

# %%
new_fea_df = feature_df[['Type_of_collision','Age_band_of_driver','Sex_of_driver',
       'Educational_level','Service_year_of_vehicle','Day_of_week','Area_accident_occured']]

oencoder2 = OrdinalEncoder()
encoded_df3 = pd.DataFrame(oencoder2.fit_transform(new_fea_df))
encoded_df3.columns = new_fea_df.columns

# %%
oencoder2.transform(new_fea_df.iloc[0:1,:]).reshape(1,-1)

# %%
joblib.dump(oencoder, "ordinal_encoder.joblib")

# %%
joblib.dump(oencoder2, "ordinal_encoder2.joblib")

# %%
final_df = pd.concat([encoded_df2,feature_df[['Hour_of_Day','Number_of_vehicles_involved','Number_of_casualties']]], axis=1)
final_df.head()

# %%
mi_calc2 = mutual_info_classif(final_df, y_en, random_state=42)
mi_df2 = pd.DataFrame({'Columns':final_df.columns, 'MI_score':mi_calc2})
mi_df2_sorted = mi_df2.sort_values(by='MI_score',ascending=False)
mi_df2_sorted

# %%
cols_list = list(mi_df2_sorted['Columns'][:10])
final_df2 = final_df[cols_list]

# %%
s_final_df = pd.concat([feature_df[['Number_of_vehicles_involved','Number_of_casualties','Hour_of_Day']],encoded_df3], axis=1)
s_final_df.head()

# %%
# train and test split and building baseline model to predict target features
X_trn2, X_tst2, y_trn2, y_tst2 = train_test_split(s_final_df, y_en, test_size=0.2, random_state=42)

# modelling using random forest baseline
rf = RandomForestClassifier(n_estimators=700, max_depth=20, random_state=42)
rf.fit(X_trn2, y_trn2)

# predicting on test data
predics2 = rf.predict(X_tst2)

# %%
X_trn2.info()

# %%
newd = X_tst2.sample(10)
sampl_arr = np.array(newd.iloc[0]).reshape(1,-1)
sampl_arr

# %%
classif_re2 = classification_report(y_tst2,predics2)
print(classif_re2)

# %%
f1score2 = f1_score(y_tst2,predics2, average='weighted')
print(f1score2)

# %%
joblib.dump(rf, "RTA_MODEL.joblib", compress=9)

# %% [markdown]
# ## Points to note:

# %%
# -Handled missing values by replacing them as aUnknown, assuming those information might not have found during investigation
# -One-Hot encoding using pandas get_dummies
# -Feature selection using chi2 statistic - Selected 50 features from 106
# -Upsampling using SMOTENC for categorical(nominal) features
# -Baseline modelling using RF with default hyper parameters - (Test f1_score - 61%)
# -Tuned hyper parameters with n_estimaters = 800 and max_depth = 20 - (Test f1_Score - 88%)

# %%



