import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as sts
from sklearn.feature_selection import mutual_info_classif
import warnings
from sklearn.metrics import mutual_info_score


warnings.filterwarnings("ignore")

def skip99(data):
    data=data.replace(999,99)
    data=data.replace(9999,99)
    data=data.replace(89,99)
    data=data.replace(989,99)
    data=data.replace(9989,99)

    return data

def blank(data):
    df_cleaned=data.replace(998,98)
    df_cleaned=data.replace(9998,98)

    num_rows_with_98 = (df_cleaned == 98).any(axis=1).sum()

    for col in df_cleaned.columns:
        if (df_cleaned[col] == 98).sum() > 22000:  #eigentlich könnte man hier die nans dazunehmen
            del df_cleaned[col]
    
    return df_cleaned

def clean_data(data):
    
    #Alle Fragen zu Drogenkonsum
    df_new = data.loc[:,'CIGEVER':'MMFGIVE']
    df_new.drop(df_new.loc[:,'PREGNANT':'YMDEIMUDPY'].columns, axis='columns', inplace=True)
    print("CIGO",df_new["CIGOFRSM"].unique())
    #df_work=data.loc[:,('QUESTID2','CADRLAST':'MMFGIVE')]

    #df_new=pd.merge(df_new, data.loc[:,'CADRLAST':'MMFGIVE'],how='left', on='QUESTID2') Fehler weil Questid2 nicht im Dataframe vohanden ist

    #Alles inklusive Mental Health Einteilung
    df_new['Mental_health_status'] = data['MI_CAT_U'] 

    #Übersicht neues Dataset
    print('Shape of dataset: ',df_new.shape)
    print('++++Columns Index++++')
    print(df_new.columns)
    print('++++Datentypen++++')
    print(df_new.dtypes)
    print('+++++Head of Dataset+++++')
    print(df_new.head(10))
    print('Number of duplicated rows:',df_new.duplicated().sum()) #duplicate rows

    #Alle Zeilen löschen mit NA in Mental Health !!wie viele? brauchen wir auch zum argumentieren für die Arbeit!!
    df_cleaned = df_new.dropna(subset=['Mental_health_status'])
    print('Number of deleted rows:', len(df_new)-len(df_cleaned))
    print("CIGO",df_cleaned["CIGOFRSM"].unique())
    missing_values_per_column = df_cleaned.isna().sum().sort_values(ascending=False)
    print('++++Number of missing values per Column++++')
    print(missing_values_per_column)

    num_columns_not_zero = sum([1 for value in missing_values_per_column if value > 0])
    print("Number of columns with missing values:", num_columns_not_zero) #100 von 1756 Spalten enthalten leere Zeilen 
    df_cleaned=df_cleaned.dropna(axis=1) #Drop alle Spalten, in denen Werte fehlen
    #macht das sinn? wird hier auch die spalte gedropped wenn nur ein value fehlt macht dann nicht reihe mehr sinn
    
    print('Are there still missing values:', df_cleaned.isna().any().any()) 
    
    #alle Spalten löschen die mehr als  2/3 Blank angaben haben
    df_cleaned=blank(df_cleaned)
    #allen skip angaben die Nummer 99 zuordnern
    df_cleaned=skip99(df_cleaned)
    
    #hier noch drüber nachdenken ob NANs und blanks dasselbe sind und ob man das so löschen möchte

    print('+++++Shape finales Dataset+++++')
    print(df_cleaned.shape) #(42739, 1632) verbliebenes Dataset davor (56136, 1756): 13397 rows and 124 columns deleted

    return df_cleaned

def replace_data(data_name):
    data_name['HALLUCEVR']=data_name['HALLUCEVR'].replace(91,2)
    data_name['INHALEVER']=data_name['INHALEVER'].replace(91,2)
    data_name['CRKEVER']=data_name['CRKEVER'].replace(91,2)
    data_name['PNRANYLIF']=data_name['PNRANYLIF'].replace(5,1)
    data_name['TRQANYLIF']=data_name['TRQANYLIF'].replace(5,1)
    data_name['STMANYLIF']=data_name['STMANYLIF'].replace(5,1)
    data_name['SEDANYLIF']=data_name['SEDANYLIF'].replace(5,1)
    data_name['PNRNMLIF']=data_name['PNRNMLIF'].replace(5,1)
    data_name['TRQNMLIF']=data_name['TRQNMLIF'].replace(5,1)
    data_name['STMNMLIF']=data_name['STMNMLIF'].replace(5,1)
    data_name['SEDNMLIF']=data_name['SEDNMLIF'].replace(5,1)
    data_name['PNRNMLIF']=data_name['PNRNMLIF'].replace(91,2)
    data_name['TRQNMLIF']=data_name['TRQNMLIF'].replace(91,2)
    data_name['STMNMLIF']=data_name['STMNMLIF'].replace(91,2)
    data_name['SEDNMLIF']=data_name['SEDNMLIF'].replace(91,2)
    
data=pd.read_csv('../data/NSDUH-2019.tsv', sep='\t')
df_cleaned=clean_data(data)

#Visualisieren
#Ob jemals Drogen konsumiert wurden bzw. Abhänigkeit von Drogen
drug_data=df_cleaned[['CIGEVER','ALCEVER','MJEVER','COCEVER','CRKEVER','HEREVER','HALLUCEVR','INHALEVER','METHAMEVR','PNRANYLIF','TRQANYLIF','STMANYLIF','SEDANYLIF','PNRNMLIF','TRQNMLIF','STMNMLIF','SEDNMLIF']]
replace_data(drug_data)
drug_data_with_ment = df_cleaned[['CIGEVER','ALCEVER','MJEVER','COCEVER','CRKEVER','HEREVER','HALLUCEVR','INHALEVER','METHAMEVR','PNRANYLIF','TRQANYLIF','STMANYLIF','SEDANYLIF','PNRNMLIF','TRQNMLIF','STMNMLIF','SEDNMLIF','Mental_health_status']] 
replace_data(drug_data_with_ment) 

melted_drug_data=drug_data.melt(var_name='Column', value_name='Value')
print(len(melted_drug_data))
filtered_drug_data=melted_drug_data[melted_drug_data['Value'].isin([1,2])] #gibt hier auch noch andere wie don't know und refused
print(len(filtered_drug_data))

drug_dependet=data[['DNICNSP', 'DEPNDALC', 'DEPNDMRJ', 'DEPNDCOC', 'DEPNDHER', 'DEPNDPYHAL','DEPNDPYINH','DEPNDPYMTH', 'DEPNDPYPNR','DEPNDPYTRQ','DEPNDPYSTM','DEPNDPYSED', 'DEPNDPYPSY']]
drug_dependet=drug_dependet.replace(0,2)
melted_drug_depended=drug_dependet.melt(var_name='Column', value_name='Value')
filtered_drug_depended= melted_drug_depended[melted_drug_depended['Value'].isin([1,2])]

list_mean_mental_health_yes = []
list_mean_mental_health_no = []
count=0
for col in drug_data_with_ment.iloc[:, :-1]:
    mean_mental_health_yes = drug_data_with_ment.loc[drug_data_with_ment[col].isin([1]), 'Mental_health_status'].mean()
    mean_mental_health_no = drug_data_with_ment.loc[drug_data_with_ment[col].isin([2]), 'Mental_health_status'].mean()
    list_mean_mental_health_yes.append(mean_mental_health_yes)
    list_mean_mental_health_no.append(mean_mental_health_no)

ment_drug_yes = pd.DataFrame({'means':list_mean_mental_health_yes,'drugs':['CIGEVER','ALCEVER','MJEVER','COCEVER','CRKEVER','HEREVER','HALLUCEVR','INHALEVER','METHAMEVR','PNRANYLIF','TRQANYLIF','STMANYLIF','SEDANYLIF','PNRNMLIF','TRQNMLIF','STMNMLIF','SEDNMLIF']})
ment_drug_no = pd.DataFrame({'means':list_mean_mental_health_no,'drugs':['CIGEVER','ALCEVER','MJEVER','COCEVER','CRKEVER','HEREVER','HALLUCEVR','INHALEVER','METHAMEVR','PNRANYLIF','TRQANYLIF','STMANYLIF','SEDANYLIF','PNRNMLIF','TRQNMLIF','STMNMLIF','SEDNMLIF']})

#Plot allgemein Druge Usage
fig1= sns.histplot(data=filtered_drug_data, x='Column', hue='Value', multiple="stack") #wär hier auch noch cool vielleicht dont know und so zu sehen also vielleicht eher melted_drug_data statt die filtered version
plt.xticks(rotation=45)
plt.legend(title='Have you ever used...', labels=['Yes','No']) #hier irgendwie vertauscht siehe plot (alkohol macht zb keinen sinn dass ja und nein so)
plt.xlabel('Different drugs')
plt.ylabel('Number of people')
plt.tight_layout()
plt.savefig('../output/drug_use.png')
plt.close()

#Plot dependency
fig2= sns.countplot(data=filtered_drug_depended, x='Column',) #stimmt irgendwas nicht kann nicht sein dass alle leute drogen nehmen
plt.xticks(rotation=45)
plt.xlabel('Different drugs')
plt.ylabel('Number of people')
plt.tight_layout()
plt.savefig('../output/drug_depended_use.png')
plt.close()

#Plot Menatl Health
fig3= sns.countplot(data=df_cleaned, x='Mental_health_status')
#plt.xticks(rotation=45)
plt.xlabel('Mental Health Status')
plt.ylabel('Number of people')
plt.tight_layout()
plt.savefig('../output/mental_health_status.png')

#Plot mental health per drug
fig4, axs = plt.subplots(2)
sns.barplot(data=ment_drug_no,x='drugs',y='means', ax=axs[0]) #hier stimmt auch irgendwas nicht da der durchschnitt über 4 ist dabei geht die scala nur bis 3
axs[0].set_title('Mental Health if no drugs')
axs[0].tick_params(axis='x', rotation=45)


sns.barplot(data=ment_drug_yes,x='drugs',y='means', ax=axs[1]) #hier anders falsch
axs[1].set_title('Mental Health if drug taken')
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig('../output/mental_health_status_vs_drugtypes.png')
plt.close()



'''
**************************************************************************************************************************************
Feature Selection
**************************************************************************************************************************************
'''
#chi test geht nicht wegen enormer sample size --> gibt unglaublich kleine p-values
def contingencytable(effect, intervention):
    global counter
    counter+=1
    contab=pd.crosstab(intervention, effect)
    """
    if counter<=15:
        print(contab)
    """
    return contab
def chisquare(df, values, label):
    d={}
    for var in values:
        table=contingencytable(df[label], df[var])
        #print(table)
        stats=[sts.chi2_contingency(table).pvalue, sts.chi2_contingency(table).statistic]
        d[var]=stats
    return d
def featureanalysis(d):
    SignificantFeatures=[]
    for key, value in d.items():
        if value[0]<SignificanceLevel:
            SignificantFeatures.append(key)
    return SignificantFeatures

def mutualinfo(values, label):
    mutualinfoscore=mutual_info_classif(values,label, discrete_features=True, random_state=42)
    d=dict(zip(values.columns, mutualinfoscore))
    return d
    

counter=0
#feature selection based on common sense

for var in df_cleaned.columns:
    if len(df_cleaned[var].unique())<2:
        del df_cleaned[var]
        counter+=1
print(counter, "columns lost their meaning because of the deletion of rows with NANs in Mental health status")
counter=0


#feature selection based on statistics
X  = df_cleaned.copy().drop('Mental_health_status', axis = 1)
Y  = df_cleaned['Mental_health_status']
SignificanceLevel=0.05
#
valuesofinterest=X.columns
#dictionary with p-value, statistic
chi2=chisquare(df_cleaned, valuesofinterest, 'Mental_health_status')
relevfeatures=featureanalysis(chi2)

mutifo=mutualinfo(X, Y)
multiinfsorted=dict(sorted(mutifo.items(), key=lambda x:x[1]))
multkey=list(multiinfsorted.keys())[:20]
multval=list(multiinfsorted.values())[:20]
plt.close()
figure = plt.barh(
    multkey,
    multval, 
    color='maroon',
    height=0.3,
)

plt.xlabel("Top features")
plt.ylabel("Feature importance")
plt.title("Feature importance of top 15 features normalized across 5 folds")
plt.tight_layout()
plt.savefig('../output/importance.png')
print('Fertig')
