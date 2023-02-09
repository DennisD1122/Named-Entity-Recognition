from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from datasets import load_dataset
import torch
from collections import Counter
import csv, json
import sys


csv.field_size_limit(sys.maxsize)

with open('article_text.csv') as f:
  csv_reader = csv.DictReader(x.replace('\0', '') for x in f)
  articles = list(dict(row) for row in csv_reader)

# Name of Hugging Face model to run
# For example, 'Davlan/bert-base-multilingual-cased-ner-hrl' or 'emilys/twitter-roberta-base-WNUT'
model_name = 'Davlan/bert-base-multilingual-cased-ner-hrl'

# Type of named entity we care about
# For example, 'ORG', 'corporation', or 'product' (depends on what the chosen model supports)
target_entity = 'ORG'

# Set up NER model
model_max_length = 512
max_len_single_sentence = model_max_length - 2
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name).to(device)
id2label = model.config.id2label
if model_name == 'emilys/twitter-roberta-base-WNUT':  # This model doesn't already have the labels specified, so we do so here
  id2label = {
    0: 'O',
    1: 'B-corporation',
    2: 'I-corporation',
    3: 'B-creative-work',
    4: 'I-creative-work',
    5: 'B-group',
    6: 'I-group',
    7: 'B-location',
    8: 'I-location',
    9: 'B-person',
    10: 'I-person',
    11: 'B-product',
    12: 'I-product'
  }


def get_pred_dict_partitions(input_ids):
  """
  Given inputs from tokenizer, returns the model's predictions using partition method
  """
  if model_max_length % 4 != 0:
    raise Exception(f'partition method relies on model_max_length being a multiple of 4, but model_max_length = {model_max_length}')
  # PyTorch tensor of input_ids divided into partitions of the maximum size
  # that the NER model can handle; padding added for incomplete partitions
  partitioned_input_ids = torch.empty(0, model_max_length)
  # Attention masks for each token in each partition
  attention_mask = torch.empty(0, model_max_length)
  # Index of SEP in input_ids
  last = len(input_ids) - 1
  # Start constructing partitions at the first token after CLS
  start = 1 
  while True:
    # Index of the token at which to end the current partition
    end = start + max_len_single_sentence
    # Padding tokens to add to this partition
    padding_len = max(0, end - last)
    padding = [tokenizer.pad_token_id] * padding_len
    # Construct partition with tokens in interval [start, end)
    # Add CLS, SEP, and padding tokens as well
    new_partition = [tokenizer.cls_token_id] + input_ids[start : min(end, last)] + [tokenizer.sep_token_id] + padding
    # Attend to all tokens except padding tokens
    new_attention_mask = [1] * (min(end, last) - start + 2) + [0] * padding_len
    # Add to tensors of all partitions and attention masks
    partitioned_input_ids = torch.cat((partitioned_input_ids, torch.tensor([new_partition])))
    attention_mask = torch.cat((attention_mask, torch.tensor([new_attention_mask])))
    if end >= last:
      # Reached last token
      break
    # Find next partition, which overlaps
    start += max_len_single_sentence // 2
  partitioned_input_ids = partitioned_input_ids.int().to(device)
  attention_mask = attention_mask.to(device)
  # Note that padding_len now stores the number of padding tokens in the last partition

  # Raw model predictions on all partitions
  partition_predictions = torch.nn.functional.softmax(
    model(input_ids=partitioned_input_ids, attention_mask=attention_mask).logits,
    dim=-1
  ).tolist()
  # List of raw predictions for each token after reassembling partitions
  predictions = []
  # Index of last partition
  last_part_i = len(partition_predictions) - 1
  for part_i, part_pred in enumerate(partition_predictions):
    # Since partitions overlap, take only the predictions for the tokens closest to the center of each partition
    # In general, this means the tokens in the first one-fourth and last one-fourth
    # of each partition are ignored; only the center one-half is used
    # The exceptions are the first partition, which must have all its beginning tokens used,
    # and the last partition, which must have all its ending tokens used (excluding padding)
    start = 1 if part_i == 0 else 1 + model_max_length // 4
    end = model_max_length - padding_len - 1 if part_i == last_part_i \
      else model_max_length - model_max_length // 4
    predictions.extend(part_pred[start:end])
  # Add list items corresponding to CLS and SEP tokens, although
  # there are no prediction results to assign to them
  predictions = [None] + predictions + [None]

  # List of dictionaries of predictions, converted into IOB2 format
  # Predictions of 'O' are not included
  pred_dict = []
  for i, pred in enumerate(predictions):
    if pred == None:
      continue
    entity = id2label[pred.index(max(pred))]
    if entity != 'O':
      pred_dict.append({
        'index': i,
        'entity': entity
      })
  
  return pred_dict


def get_named_entities(text):
  """
  Given string of text, returns list of named entities
  """
  # Results of running tokenizer
  inputs = tokenizer(text)
  # List associating token indices with word indices
  word_ids = inputs.word_ids()
  # Number of words
  word_count = 1 + next((elt for elt in reversed(word_ids) if elt is not None), -1)
  # List of words
  words = [
    text[inputs.word_to_chars(i)[0] : inputs.word_to_chars(i)[1]]
    for i in range(word_count)
  ]

  # Get model predictions
  pred_dict = get_pred_dict_partitions(inputs['input_ids'])

  # Dictionary associating word ids with predicted labels
  word_to_pred = {}
  for pred in pred_dict:
    word_id = word_ids[pred['index']]
    # Add label on word if no label yet
    # and prioritize target entities (but don't change 'B-' into 'I-')
    if word_id not in word_to_pred or \
      (pred['entity'][2:] == target_entity and word_to_pred[word_id] != 'B-'+target_entity):
      word_to_pred[word_id] = pred['entity']
  # Only keep target entity
  word_to_pred = {k:v for k,v in word_to_pred.items() if v[2:] == target_entity}
  
  # prev_alnum_word_ids[i] gives the id of the last alphanumeric word that appears before word i
  # Used for ignoring non-alphanumeric words
  prev_alnum_word_ids = []
  alnum_id = None
  for i in range(word_count):
    prev_alnum_word_ids.append(alnum_id)
    if words[i].isalnum():
      alnum_id = i

  # List of dictionaries representing the text spans identified as named entities
  named_entities = []
  prev_lab_alnum_word_id = None
  prev_lab_is_alnum = True  # Initialized to True in order to trigger conditional on the first loop iteration
  for word_id, label in word_to_pred.items():
    # Beginning of entity ("B-"), or an "I-" label without a preceding labeled alphanumeric word
    if label[0] == 'B' or \
        ((prev_alnum_word_ids[word_id] != prev_lab_alnum_word_id or prev_lab_alnum_word_id == None) and prev_lab_is_alnum):
      # Get character index that starts this new entity
      start = inputs.word_to_chars(word_id)[0]
      # Place a new element into list -- the actual value is set below
      named_entities.append(None)

    # Regardless of whether this word begins or is inside a named entity:
    # Update ending character index
    end = inputs.word_to_chars(word_id)[1]
    # Update entity (the last element of the entity list represents the entity that's currently being built)
    named_entities[-1] = text[start:end]

    if words[word_id].isalnum():
      prev_lab_alnum_word_id = word_id  # The index of the most recent labeled alphanumeric word
      prev_lab_is_alnum = True  # Whether the previous labeled word is alphanumeric
    else:
      prev_lab_is_alnum = False
    
  return named_entities


# Get target named entities in all articles
named_entities_by_article = []
for i, article in enumerate(articles):
  # Text of article
  text = article['text']
  # Target named entities found in current article
  named_entities = []
  start_char = 0
  while True:
    # Run NER on 10,000-character chunks because of memory constraints
    named_entities.extend(get_named_entities(text[start_char : start_char+10000]))
    start_char += 10000
    if start_char >= len(text):
      break
  print(i, named_entities)
  # Add article and results to final list
  named_entities_by_article.append({'id': article['id'], 'results': named_entities})

with open('named_entities_in_article_text.json', 'w') as f:
  json.dump(named_entities_by_article, f, indent=4)
