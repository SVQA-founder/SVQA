import csv
import json
import tqdm

def convert_to_csv(input_path, output_path):
  with open(input_path) as f:
    data = json.load(f)
  print('\033[;31mConvert to csv file ...\033[0m')
  with open(output_path, 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(['question', 'answer', 'description', 'type', 'key', 'vid_id'])
    for d in tqdm.tqdm(data):
      writer.writerow([d['question'], str(d['ans']), d['question'], str(2), str(d['id']), str(d['id'])])

def convert_total(train, test, output_path):
  with open(train) as f:
    d1 = json.load(f)
  with open(test) as f:
    d2 = json.load(f)

  print('\033[;31mConvert total to csv file ...\033[0m')
  with open(output_path, 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(['question', 'answer', 'description', 'type', 'key', 'vid_id'])
    for d in tqdm.tqdm(d1):
      writer.writerow([d['question'], str(d['ans']), d['question'], str(2), str(d['id']), str(d['id'])])
    for d in tqdm.tqdm(d2):
      writer.writerow([d['question'], str(d['ans']), d['question'], str(2), str(d['id']), str(d['id'])])


if __name__ == '__main__':
  convert_to_csv('./data/questions/train.json', './data/questions/train.csv')
  convert_to_csv('./data/questions/test.json', './data/questions/test.csv')
  convert_total('./data/questions/train.json', './data/questions/test.json', './data/questions/total.csv')