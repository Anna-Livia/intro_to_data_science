import pandas as pd
import numpy as np
import seaborn as sb

train = pd.read_csv('boites_medicaments_train.csv', encoding='utf-8', sep=';')

print(train.head(5))

train['logprix'] = np.log(train['prix'])

sb.distplot(train['logprix'], kde=False)
sb.plt.show()
sb.violinplot(y='logprix', x='tx rembours', data=train)
sb.plt.show()

temp = train['titulaires'].apply(lambda st: st.split(','))
labos = set([element for listElement in temp for element in listElement])
for lab in labos:
    train[lab] = train['titulaires'].apply(lambda x: 1 if lab in x else 0)




