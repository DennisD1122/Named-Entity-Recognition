import pandas as pd
import numpy as np
import json

df_all = pd.read_csv('companies_dataset.csv')

with open('/content/drive/My Drive/HDAG/named_entities_in_article_text_organizations.json') as f:
  contents = json.load(f)

def preprocess_name(name):
  name2 = ''
  for char in str(name):
    if char.isalpha():
      name2 += char.lower()
  return name2

entities = []
for article in contents:
  entities.extend(article['entities'])
for i, entity in enumerate(entities):
  entities[i] = entity, preprocess_name(entity)

relevant_industries = 'pharmaceuticals', 'medical devices', 'biotechnology', 'chemicals', 'nanotechnology'

df = df_all[df_all['industry'].isin(relevant_industries)]
df.insert(2, 'preprocessed_name', df['name'].apply(preprocess_name))

companies = []
for i, (entity, preprocessed_entity) in enumerate(entities):
  if i % 100 == 0: print(i)
  if any(preprocessed_name == preprocessed_entity for preprocessed_name in df['preprocessed_name'].values):
    companies.append(entity)

for e in np.unique(companies):
  print(e)
