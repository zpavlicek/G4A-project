import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data=pd.read_csv('../data/NSDUH-2019.tsv', sep='\t')

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
        if (df_cleaned[col] == 98).sum() > 22000:  
            del df_cleaned[col]
    
    return df_cleaned

def clean_data(data):
    
    #Alle Fragen zu Drogenkonsum
    df_new = data.loc[:,'CIGEVER':'NDTRNMIMPT']
    df_new=pd.merge(df_new, data.loc[:,'CADRLAST':'MMFGIVE'],how='left', on='QUESTID2')

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

    missing_values_per_column = df_cleaned.isna().sum().sort_values(ascending=False)
    print('++++Number of missing values per Column++++')
    print(missing_values_per_column)

    num_columns_not_zero = sum([1 for value in missing_values_per_column if value > 0])
    print("Number of columns with missing values:", num_columns_not_zero) #100 von 1756 Spalten enthalten leere Zeilen 
    df_cleaned=df_cleaned.dropna(axis=1) #Drop alle Spalten, in denen Werte fehlen
    
    print('Are there still missing values:', df_cleaned.isna().any().any()) 
    
    #alle Spalten löschen die mehr als  2/3 Blank angaben haben
    df_cleaned=blank(df_cleaned)
    #allen skip angaben die Nummer 99 zuordnern
    df_cleaned=skip99(df_cleaned)
    
    print('+++++Shape finales Dataset+++++')
    print(df_cleaned.shape) #(42739, 1632) verbliebenes Dataset davor (56136, 1756): 13397 rows and 124 columns deleted

    return df_cleaned

df_cleaned=clean_data(data)

#Visualisieren
#Ob jemals Drogen konsumiert wurden bzw. Abhänigkeit von Drogen
drug_data=df_cleaned[['CIGEVER','ALCEVER','MJEVER','COCEVER','CRKEVER','HEREVER','HALLUCEVR','INHALEVER','METHAMEVR','PNRANYLIF','TRQANYLIF','STMANYLIF','SEDANYLIF','PNRNMLIF','TRQNMLIF','STMNMLIF','SEDNMLIF']]
drug_data['HALLUCEVR']=drug_data['HALLUCEVR'].replace(91,2)
drug_data['INHALEVER']=drug_data['INHALEVER'].replace(91,2)
drug_data_with_ment = df_cleaned[['CIGEVER','ALCEVER','MJEVER','COCEVER','CRKEVER','HEREVER','HALLUCEVR','INHALEVER','METHAMEVR','PNRANYLIF','TRQANYLIF','STMANYLIF','SEDANYLIF','PNRNMLIF','TRQNMLIF','STMNMLIF','SEDNMLIF','Mental_health_status']]
drug_data_with_ment['HALLUCEVR']=drug_data_with_ment['HALLUCEVR'].replace(91,2)
drug_data_with_ment['INHALEVER']=drug_data_with_ment['INHALEVER'].replace(91,2) #wieso nur hier???

melted_drug_data=drug_data.melt(var_name='Column', value_name='Value')
print(len(melted_drug_data))
filtered_drug_data=melted_drug_data[melted_drug_data['Value'].isin([1,2])] #sehr viel geht hier verloren z.B. Sednmilf hat es auch 91 was never used bedeutet

melted_drug_data_with_ment=drug_data_with_ment.melt(var_name='Column', value_name='Value')
filtered_drug_data_with_ment=melted_drug_data_with_ment[melted_drug_data_with_ment['Value'].isin([1,2])] #wo bleibt der rest? mental health hat 1-4
print(len(filtered_drug_data))
drug_dependet=data[['DNICNSP', 'DEPNDALC', 'DEPNDMRJ', 'DEPNDCOC', 'DEPNDHER', 'DEPNDPYHAL','DEPNDPYINH','DEPNDPYMTH', 'DEPNDPYPNR','DEPNDPYTRQ','DEPNDPYSTM','DEPNDPYSED', 'DEPNDPYPSY']]
melted_drug_depended=drug_dependet.melt(var_name='Column', value_name='Value')
filtered_drug_depended= melted_drug_depended[melted_drug_depended['Value'].isin([1,2])] #exclude other values that are not 1 or 2

list_mean_mental_health_yes = []
list_mean_mental_health_no = []
count=0
for col in drug_data_with_ment.iloc[:, :-1]:
    count+=1
    print(count)
    mean_mental_health_yes = drug_data_with_ment.loc[drug_data_with_ment[col].isin([1]), 'Mental_health_status'].mean()
    mean_mental_health_no = drug_data_with_ment.loc[drug_data_with_ment[col].isin([2]), 'Mental_health_status'].mean()
    list_mean_mental_health_yes.append(mean_mental_health_yes)
    list_mean_mental_health_no.append(mean_mental_health_no)

ment_drug_yes = pd.DataFrame({'means':list_mean_mental_health_yes,'drugs':['CIGEVER','ALCEVER','MJEVER','COCEVER','CRKEVER','HEREVER','HALLUCEVR','INHALEVER','METHAMEVR','PNRANYLIF','TRQANYLIF','STMANYLIF','SEDANYLIF','PNRNMLIF','TRQNMLIF','STMNMLIF','SEDNMLIF']})
ment_drug_no = pd.DataFrame({'means':list_mean_mental_health_no,'drugs':['CIGEVER','ALCEVER','MJEVER','COCEVER','CRKEVER','HEREVER','HALLUCEVR','INHALEVER','METHAMEVR','PNRANYLIF','TRQANYLIF','STMANYLIF','SEDANYLIF','PNRNMLIF','TRQNMLIF','STMNMLIF','SEDNMLIF']})

#Plot allgemein Druge Usage
fig1= sns.histplot(data=filtered_drug_data, x='Column', hue='Value', multiple="stack") #haben wir hier noch eingefügt damit man was lesen kann
plt.xticks(rotation=45)
plt.legend(title='Have you ever used...', labels=['Yes','No'])
plt.xlabel('Different drugs')
plt.ylabel('Number of people')
plt.tight_layout()
plt.savefig('../output/drug_use.png')
plt.close()

#Plot dependency
fig2= sns.countplot(data=filtered_drug_depended, x='Column',)
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
sns.barplot(data=ment_drug_no,x='drugs',y='means', ax=axs[0])
axs[0].set_title('Mental Health if no drugs')
axs[0].tick_params(axis='x', rotation=45)


sns.barplot(data=ment_drug_yes,x='drugs',y='means', ax=axs[1])
axs[1].set_title('Mental Health if drug taken')
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig('../output/mental_health_status_vs_drugtypes.png')

print('Fertig')
