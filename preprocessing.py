import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data=pd.read_csv('NSDUH-2019.tsv', sep='\t')

#Alle Fragen zu Drogenkonsum
df_new = data.loc[:,'CIGEVER':'MMFGIVE']

#Alles inklusive Mental Health Einteilung
df_new['Mental_health_status'] = data['MI_CAT_U']
print(data['MI_CAT_U'].unique())

#Übersicht neues Dataset
print(df_new.shape)
print(df_new.columns)
print(df_new.dtypes)
print(df_new.head(10))
print(df_new.duplicated().sum()) #duplicate rows

#Alle Zeilen löschen in denen nur 640 Fragen oder weniger beantwortet wurden
count_skip=df_new.eq(99).sum(axis=1)
df_cleaned = df_new[count_skip <= 640] #der count bezieht sich auf wie viele nicht beantwortet wurden (mit 99 im dataset gekennzeichnet), also je kleiner desto besser

#Alle Zeilen löschen mit NA in Mental Health
df_cleaned = df_new.dropna(subset=['Mental_health_status'])

#Visualisieren
#Ob jemals Drogen konsumiert wurden bzw. Abhänigkeit von Drogen
drug_data=df_cleaned[['CIGEVER','ALCEVER','MJEVER','COCEVER','CRKEVER','HEREVER','HALLUCEVR','INHALEVER','METHAMEVR','PNRANYLIF','TRQANYLIF','STMANYLIF','SEDANYLIF','PNRNMLIF','TRQNMLIF','STMNMLIF','SEDNMLIF']]
drug_data['HALLUCEVR']=drug_data['HALLUCEVR'].replace(91,2)
drug_data['INHALEVER']=drug_data['INHALEVER'].replace(91,2)
drug_data_with_ment = df_cleaned[['CIGEVER','ALCEVER','MJEVER','COCEVER','CRKEVER','HEREVER','HALLUCEVR','INHALEVER','METHAMEVR','PNRANYLIF','TRQANYLIF','STMANYLIF','SEDANYLIF','PNRNMLIF','TRQNMLIF','STMNMLIF','SEDNMLIF','Mental_health_status']]
drug_data_with_ment['HALLUCEVR']=drug_data_with_ment['HALLUCEVR'].replace(91,2)
drug_data_with_ment['INHALEVER']=drug_data_with_ment['INHALEVER'].replace(91,2)

melted_drug_data=drug_data.melt(var_name='Column', value_name='Value')
filtered_drug_data=melted_drug_data[melted_drug_data['Value'].isin([1,2])]

melted_drug_data_with_ment=drug_data_with_ment.melt(var_name='Column', value_name='Value')
filtered_drug_data_with_ment=melted_drug_data_with_ment[melted_drug_data_with_ment['Value'].isin([1,2])]

drug_dependet=data[['DNICNSP', 'DEPNDALC', 'DEPNDMRJ', 'DEPNDCOC', 'DEPNDHER', 'DEPNDPYHAL','DEPNDPYINH','DEPNDPYMTH', 'DEPNDPYPNR','DEPNDPYTRQ','DEPNDPYSTM','DEPNDPYSED', 'DEPNDPYPSY']]
melted_drug_depended=drug_dependet.melt(var_name='Column', value_name='Value')
filtered_drug_depended= melted_drug_depended[melted_drug_depended['Value'].isin([1,2])] #exclude other values that are not 1 or 2

list_mean_mental_health_yes = []
list_mean_mental_health_no = []
for col in drug_data_with_ment.iloc[:, :-1]:
    mean_mental_health_yes = drug_data_with_ment.loc[drug_data_with_ment[col].isin([1]), 'Mental_health_status'].mean()
    mean_mental_health_no = drug_data_with_ment.loc[drug_data_with_ment[col].isin([2]), 'Mental_health_status'].mean()
    list_mean_mental_health_yes.append(mean_mental_health_yes)
    list_mean_mental_health_no.append(mean_mental_health_no)

ment_drug_yes = pd.DataFrame({'means':list_mean_mental_health_yes,'drugs':['CIGEVER','ALCEVER','MJEVER','COCEVER','CRKEVER','HEREVER','HALLUCEVR','INHALEVER','METHAMEVR','PNRANYLIF','TRQANYLIF','STMANYLIF','SEDANYLIF','PNRNMLIF','TRQNMLIF','STMNMLIF','SEDNMLIF']})
ment_drug_no = pd.DataFrame({'means':list_mean_mental_health_no,'drugs':['CIGEVER','ALCEVER','MJEVER','COCEVER','CRKEVER','HEREVER','HALLUCEVR','INHALEVER','METHAMEVR','PNRANYLIF','TRQANYLIF','STMANYLIF','SEDANYLIF','PNRNMLIF','TRQNMLIF','STMNMLIF','SEDNMLIF']})

#Plot allgemein Druge Usage
fig1= sns.countplot(data=filtered_drug_data, x='Column', hue='Value')
plt.xticks(rotation=45)
plt.legend(title='Have you ever used...', labels=['Yes','No'])
plt.xlabel('Different drugs')
plt.ylabel('Number of people')
plt.tight_layout()
plt.savefig('drug_use.png')
plt.close()

#Plot dependency
fig2= sns.countplot(data=filtered_drug_depended, x='Column',)
plt.xticks(rotation=45)
plt.xlabel('Different drugs')
plt.ylabel('Number of people')
plt.tight_layout()
plt.savefig('drug_depended_use.png')
plt.close()

#Plot Menatl Health
fig3= sns.countplot(data=df_cleaned, x='Mental_health_status')
#plt.xticks(rotation=45)
plt.xlabel('Mental Health Status')
plt.ylabel('Number of people')
plt.tight_layout()
plt.savefig('mental_health_status.png')

#Plot mental health per drug
fig4, axs = plt.subplots(2)
sns.barplot(data=ment_drug_no,x='drugs',y='means', ax=axs[0])
axs[0].set_title('Mental Health if no drugs')
axs[0].tick_params(axis='x', rotation=45)


sns.barplot(data=ment_drug_yes,x='drugs',y='means', ax=axs[1])
axs[1].set_title('Mental Health if drug taken')
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig('mental_health_status_vs_drugtypes.png')

print('Fertig')
