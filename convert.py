import pickle 
import csv 
import numpy as np
import jieba
import json
embedding = []
word2index = {}

bert_dic = {}


with open("datasets/OCNLI/train.50k.json","r") as f:
    for k,line in enumerate(f):
        loaded_example = json.loads(line)
        sentence1 = loaded_example['sentence1']
        sentence2 = loaded_example['sentence2']

        seg_list = jieba.cut(sentence1)
        for text in seg_list:
            text = text.strip()
            if text not in bert_dic:
                bert_dic[text] = []

        seg_list = jieba.cut(sentence2)
        for text in seg_list:
            text = text.strip()
            if text not in bert_dic:
                bert_dic[text] = []
    print(f"total voc:{len(bert_dic)}")

# with open("vocab.txt","r") as f:
#     txt = f.readlines()
#     for i in txt:
#         i = i.strip()
#         bert_dic[i] = []


with open("merge_sgns_bigram_char300.txt", encoding='utf-8', errors='ignore') as f:
    found = 0
    print("hello")
    # f.read()
    first_line = True
    # 讀取 CSV 檔案內容
    # 以冒號分隔欄位，讀取檔案內容
    # rows = csv.reader(f, delimiter=' ')
    index = -1
    # 以迴圈輸出每一列
    for line in f:
        if first_line:
            first_line = False
            continue
        index+=1
        tokens = line.rstrip().split(' ')
        temp = tokens
        word = temp[0]
        if word in bert_dic:
            found+=1
            word2index[word] = len(word2index)
            embedding.append(temp[1:])
            # vecr = temp[1:]
            # words[word] = vecr
        print(index,end ="\r")
    print(f"found in word embedding:{found}")
embedding = np.array(embedding)
np.save('paragram_words', embedding)
np.save("wordlist_words",word2index ,allow_pickle=True)

