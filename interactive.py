"Evaluate the model"""
import os
import re
import nltk
import torch
import random
import logging
import argparse
import numpy as np
import utils as utils
import pandas as pd
from metrics import get_entities
from data_loader import DataLoader
from SequenceTagger import BertForSequenceTagging

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='msra', help="Directory containing the dataset")
parser.add_argument('--seed', type=int, default=23, help="random seed for initialization")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def interAct(model, data_iterator, params, mark='Interactive', verbose=False):
    """Evaluate the model on `steps` batches."""
    # set model to evaluation mode
    model.eval()
    idx2tag = params.idx2tag

    batch_data, batch_token_starts = next(data_iterator)
    batch_masks = batch_data.gt(0)
        
    batch_output = model((batch_data, batch_token_starts), token_type_ids=None, attention_mask=batch_masks)[0]  # shape: (batch_size, max_len, num_labels)
    batch_output = batch_output.detach().cpu().numpy()
    
    pred_tags = []
    pred_tags.extend([[idx2tag.get(idx) for idx in indices] for indices in np.argmax(batch_output, axis=2)])
    
    return(get_entities(pred_tags))


def bert_ner_init():
    args = parser.parse_args()
    tagger_model_dir = 'experiments/' + args.dataset

    # Load the parameters from json file
    json_path = os.path.join(tagger_model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Use GPUs if available
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set the random seed for reproducible experiments
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    params.seed = args.seed

    # Set the logger
    utils.set_logger(os.path.join(tagger_model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Loading the dataset...")

    # Initialize the DataLoader
    data_dir = 'data/' + args.dataset
    if args.dataset in ["conll"]:
        bert_class = 'bert-base-cased'
    elif args.dataset in ["msra"]:
        bert_class = 'bert-base-chinese'

    data_loader = DataLoader(data_dir, bert_class, params, token_pad_idx=0, tag_pad_idx=-1)

    # Load the model
    model = BertForSequenceTagging.from_pretrained(tagger_model_dir)
    model.to(params.device)

    return model, data_loader, args.dataset, params

def BertNerResponse(model, queryString):    #结果是一个列表. 每次apppen一个元组, 元素分别是检测的元素以及对应的标识.[('Jason Momoa', 'PER'), ('Lisa Bonet', 'PER'), ('Instagram', 'ORG')]
    model, data_loader, dataset, params = model
    if dataset in ['msra']:
        queryString = [i for i in queryString]
    elif dataset in ['conll']:
        queryString = nltk.word_tokenize(queryString)


    with open('data/' + dataset + '/interactive/sentences.txt', 'w') as f:
        f.write(' '.join(queryString))

    inter_data = data_loader.load_data('interactive')
    inter_data_iterator = data_loader.data_iterator(inter_data, shuffle=False)
    result = interAct(model, inter_data_iterator, params)
    res = []
    for item in result:
        if dataset in ['msra']:
            res.append((''.join(queryString[item[1]:item[2]+1]), item[0]))
        elif dataset in ['conll']:
            res.append((' '.join(queryString[item[1]:item[2]+1]), item[0]))
    return res


Dic = {}

Dic['OCC'] = ['Text', 'Occupation', 'a job']
Dic['PER'] = ['Text', 'Person', 'name of person']
Dic['ORG'] = ['Text', 'Organization', 'organizations such as private company and goverment']
Dic['LOC'] = ['Text', 'Location', 'a place']
Dic['Disease'] = ['Text', 'Disease', 'an illness affecting humans, animals or plants, often caused by infection']

Dic['Email'] = ['Text', 'Email address', "an individual name that you use to receive email on the internet"]
Dic['Ip'] = ['Text', 'Ip address', 'identifies a particular computer connected to the Internet']
Dic['Domain'] = ['Text', 'Domain name', 'a name which identifies a website or group of websites on the Internet']
Dic['Phone'] = ['Text', 'Phone Number', 'a series of digit numbers that you use to call a particular person on the telephone']
Dic['Credit card'] = ['Text', 'Credit card ', 'a small plastic card that you use to buy things now and pay for them later']
Dic['Passport'] = ['Text', 'Passport', 'an official document that identifies you as a citizen of a particular country']

def RegularMatch(str):

    regular_result = {}
    re_email = re.compile(r'([a-zA-Z0-9_-]+@[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+)+')
    email_list = re_email.findall(str)

    re_ip_address = re.compile(r'((?:[0,1]?\d{1,2}|2(?:[0-4][0-9]|5[0-5]))(?:\.(?:[0,1]?\d{1,2}|2(?:[0-4][0-9]|5[0-5]))){3})')
    ip_address_list = re_ip_address.findall(str)

    re_domain_name = re.compile(r'[^/"><\.]{3,14}\.com|[^/"><\.]{4,}\.cn|[^/"><\.]{3,14}\.net\.cn') #domain有可能检测出邮箱地址. 因此匿名化处理时需要做一些裁剪.
    domain_name_list = re_domain_name.findall(str)

    re_phone_number = re.compile(r'1(?:[358][0-9]|4[579]|66|7[0135678]|9[89])[0-9]{8}')
    phone_list = re_phone_number.findall(str)

    re_credit_card = re.compile(r'^4[0-9]{12}(?:[0-9]{3})?') #4155279860457
    card_list = re_credit_card.findall(str)

    re_pass_port = re.compile(r'(?:E\d{8})|E[A-Za-z]\d{7}|(?:G\d{8})|(?:H\d{8})|(?:HJ\d{7})|(?:K\d{8})|(?:KJ\d{7})|(?:MB\d{7})') #G28233515
    pass_port_list = re_pass_port.findall(str)
    real_pass_port_list = []
    for value in pass_port_list:
        if value != '':
            real_pass_port_list.append(value)

    regular_result['Email'] = email_list
    regular_result['Ip'] = ip_address_list
    regular_result['Domain'] = domain_name_list
    regular_result['Phone'] = phone_list
    regular_result['Credit Card'] = card_list
    regular_result['Passport'] = real_pass_port_list
    return regular_result

def regulartrans(result, str): #此时result的格式: result是一个Dic.
    #将str中检测出来的数据(此时用result保存在), 最后替换掉str中的原始数据. 然后返回字符串.
    if len(result) == 0:
        return str
    Email_list = result['Email']
    for key, value in result.items():
        for values in value:
            if key == 'Domain' and values in Email_list:
                pass
            str = str.replace(values, '<mark>' + Dic[key][1].upper() + '</mark>')
    return str


def trans(result, str):#此时result的格式: 一个list[],其中中间的元素为元组, 可能有多个.
    #htmlstr = str
    if len(result) == 0:
        return str
    for tuples in result:# tuples[0]是str中的对应元素, tuples[1]是Dic对应的元素.
        str = str.replace(tuples[0], '<mark>' + Dic[tuples[1]][1].upper() + '</mark>')
    return str
# def transhtml(result, regular_result, query):
#     for word in result:

#query = "Peter loves computer. His email_address is peter@gmail.com. He creates his own website www.peter.com. This website's ip_address is 202.192.7.1. This year, he got his own passport G28233515. Every one can call 18382312429 to find him"
#def main():
def sensitive_data_detection(query):

    model = bert_ner_init()
    word_list = pd.DataFrame()

    kind_list = ['Value', 'Type', 'Class', 'Description']
    file_name = "results.csv"
    #while True:
        #query = "Jason Momoa and Lisa Bonet released a joint statement posted on his verified Instagram account Wednesday, announcing they are ending their marriage." # "They are both actor" + "The pair reportedly met at a jazz club in 2005, well before Momoa became a famous actor playing Khal Drogo on Game of Thrones"
    #query = "Peter lives in HongKong. His commonly used e-mail is peterforever@gmail.com. He works as a teacher who is loved and respected by students. In his causal time, he loves web-programming. His personal website is www.peter23forever.com. or you can enter 207.24.1.3 to get access to it."
    result = BertNerResponse(model, query)

    query = trans(result, query)

    regular_result = RegularMatch(query)

    query = regulartrans(regular_result, query)

    #html_query = transhtml(result, regular_result, query)
        #for i in range(len(query)):
        #    if i % 60 == 0:
        #        print()
        #    print(query[i], end="")

        #print(query)
        #query = "They are both actor."
        #query = "The pair reportedly met at a jazz club in 2005, well before Momoa became a famous actor playing Khal Drogo on Game of Thrones"
        #query = "They are both actor"
        #if query == 'exit':
        #    break

        #string =
        # 此时，将result转换成双重list. Value, Type, Arri, Describition.
        #遍历result.

    for regular_key, regular_values in regular_result.items():
        for value in regular_values:
            data = []
            data.append(value)
            for word in Dic[regular_key]:
                data.append(word)
            word_list = word_list.append([[data[i] for i in range(len(data))]], ignore_index=True)

    for ans in result:
        data = []
        data.append(ans[0])
        for word in Dic[ans[1]]:
            data.append(word)
        word_list = word_list.append([[data[i] for i in range(len(data))]], ignore_index=True)

    word_list.columns = kind_list
    word_list.to_csv(file_name, encoding="utf-8")


        #word_list是一个双重列表.
    return query


from flask import Flask, render_template,request
from interactive import BertForSequenceTagging, BertNerResponse
app = Flask(__name__)

@app.route('/', methods = ['GET','POST'])
def index():
    #model = bert_ner_init()
    ans = ""
    new_str = ""
    students1 = [98,95,92,99,78]
    student2 = [95,85,68,48,45]
    student = [students1, student2]
    if request.method == 'POST':
        new_str = request.form['new_str']
        #ans = BertNerResponse(model, new_str)
        ans = sensitive_data_detection(new_str)

        # 在这儿重新组装 ans.# 试一下，看不能高亮
        ans = "<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" + ans
        #ans ='<mark>' + ans
    print(ans)
    return render_template("index.html", get_str=ans, ori_str=new_str, E_data=student)


if __name__ == "__main__":
    app.run(debug=True)
#if __name__ == '__main__':
#    main()


    

