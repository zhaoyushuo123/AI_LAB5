import os

data_local = "data/实验五数据/实验五数据/"
train_txt = [w.strip() for w in open(data_local+"train.txt",'r').readlines()][1:]
test_without_label_txt = [w.strip() for w in open(data_local+"test_without_label.txt",'r').readlines()][1:]


length_of_train_txt=len(train_txt)
length_of_valid=int(length_of_train_txt/5)
print(length_of_train_txt)
print(length_of_valid)
train = train_txt[length_of_valid:]
print(len(train))
valid = train_txt[:length_of_valid]
print(len(valid))
labels = ['negative', 'neutral', 'positive']

def save_in_txt(save_text,save_image,datasets):
    pass
    ft = open(save_image,'w',encoding='utf-8')
    with open(save_text,'w',encoding='utf-8') as f:
        for t in datasets:
            text_index,label = t.split(',')
            label = str(labels.index(label))
            content = open(f'data/实验五数据/实验五数据/data/{text_index}.txt', 'r', encoding='gbk', errors='ignore').read()
            f.write(content.strip() + '\t' + label + '\n')
            ft.write(t + '\n')
        ft.close()

save_in_txt("new_data/text/train.txt","new_data/image/train.txt",train)
save_in_txt("new_data/text/valid.txt","new_data/image/valid.txt",valid)

f = open('new_data/image/test.txt', 'w', encoding='utf8')
with open('new_data/text/test.txt', 'w', encoding='utf8') as f:
    for t in test_without_label_txt:
        text_index, _ = t.split(',')
        label = "0"
        content = open(f'data/实验五数据/实验五数据/data/{text_index}.txt', 'r', encoding='gbk', errors='ignore').read()
        f.write(content.strip() + '\t' + label + '\n')
        f.write(t.replace('null','negative') + '\n')

    f.close()


