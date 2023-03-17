import os
import torch
import numpy as np
import pandas as pd
import difflib
import pickle
import nltk
from tqdm import tqdm
import argparse
import requests
import json as jsonn
import random

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn import functional as F

from dataloader import DatasetMetaQA, DataLoaderMetaQA
from model import RelationExtractor

from SPARQLWrapper import SPARQLWrapper, JSON

from sanic import Sanic
from sanic.response import json, text
from sanic_cors import CORS, cross_origin

from sanic_ipware import get_client_ip

from autocorrect import Speller
from polyfuzz import PolyFuzz
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import logging
import time


nltk.download('punkt')

os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = Sanic(name="prodsearchdemo")
CORS(app)

app.static("/static", "./images")
device = "cpu"

model = torch.load(
    "./fashion_nerda_mbert_model_V2.pt", map_location=torch.device(device)
)  # , map_location =torch.device('cpu'))

kg_data_df = pd.read_csv("./kg_prod_df_V2.csv")
fuzzy_model = PolyFuzz("TF-IDF")
sparql_conn = SPARQLWrapper("http://graph_db:7200/repositories/FashNew")

url = "https://revgen.reverieinc.com/translate"
headers = {
    "Content-Type": "application/json",
    #'src_lang': "hi",
    "tgt_lang": "en",
    "cache-control": "no-cache",
}

logging.basicConfig(
    filename="fashionSearchDemo_AJIO.log",
    format="%(asctime)s %(message)s",
    filemode="a",
)
# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)

category_dict = {}
for category_type in list(kg_data_df.articleType.unique()):
    pid = random.sample(
        list(kg_data_df[kg_data_df.articleType == str(category_type)].Id), k=1
    )[0]
    category_dict[category_type] = (
        "https://staging-jio.reverieinc.com/get_kgqa_static/" + str(pid) + ".jpg"
    )

lang_dict = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Tamil": "ta",
    "Bengali": "bn",
    "Gujarati": "gu",
    "Marathi": "mr",
}

#predict category of the text
def PredictText(query):
    # print("queryyyy",query)
    predict = model.predict_text(query)
    # print("predict",predict)
    return predict

#translate the text to English if input text is in any other language
def translate_query(query, src_lang):
    url = "https://revapi.reverieinc.com/translate"

    payload = jsonn.dumps(
        {
            "data": [query],
            "n_best": 2,
            "mask": True,
            "mask_terms": ["Reverie Language Technologies"],
            "segment_after": 0,
        }
    )
    headers = {
        "REV-API-KEY": "503b66fe4dc6695a92043a22198ad1f870d1ee15",
        "REV-APP-ID": "rev.voicesuite",
        "REV-APPNAME": "localization",
        "src_lang": src_lang,
        "tgt_lang": "en",
        "Content-Type": "application/json",
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    return jsonn.loads(response.text)["result"][0][0]


def get_prod_info(prod_id_list):
    #     begin = time.time()
    prod_info_dict = []
    #     print("prod_id_list",prod_id_list)
    for pid in prod_id_list:
        prd_dict = {}
        prd_row = kg_data_df[kg_data_df.Id == pid]

        prd_dict["Id"] = pid
        prd_dict["price"] = str(prd_row.iloc[0]["price"])
        prd_dict["brandName"] = str(prd_row.iloc[0]["brandName"])
        prd_dict["discountedPrice"] = str(prd_row.iloc[0]["discountedPrice"])
        prd_dict["gender"] = str(prd_row.iloc[0]["gender"])
        prd_dict["color"] = str(prd_row.iloc[0]["color"])
        prd_dict["usage"] = str(prd_row.iloc[0]["usage"])
        prd_dict["isEMIEnabled"] = str(prd_row.iloc[0]["isEMIEnabled"])
        prd_dict["isReturnable"] = str(prd_row.iloc[0]["isReturnable"])
        prd_dict["isExchangeable"] = str(prd_row.iloc[0]["isExchangeable"])
        prd_dict["img_url"] = (
            "https://staging-jio.reverieinc.com/get_kgqa_static/" + str(pid) + ".jpg"
        )

        prod_info_dict.append(prd_dict)
    #     end = time.time()
    #     print(f"Total runtime of the program in get_prod_info is {end - begin}")
    return prod_info_dict


f = open("./fashDict.json", "r")
fash_data = jsonn.load(f)

# Converts misspelled words to the correct words
def process_query_v3(query):
   # begin = time.time()
    d = {}
    (
        string,
        string2,
        _color_,
        _artType_,
        _gender_,
        _brandName_,
        _season_,
        _usage_,
        _age_,
    ) = ("", "", "", "", "", "", "", "", "")

    pred = PredictText(query)

    for i in range(len(pred[1][0])):
        if pred[1][0][i] in d:
            d[pred[1][0][i]].append(pred[0][0][i])
        else:
            d[pred[1][0][i]] = [pred[0][0][i]]

    for i in d.keys():
        if i != "other" and i != "articletype":
            string += " " + " ".join(d[i])
        elif i == "articletype":
            string2 += " " + " ".join(d[i])

    string = string + string2
    str_split = string.split()

    for item in str_split:
        if d.get("color") and item in d.get("color"):
            _color_ = " ".join(d.get("color"))
            ratios = {i: fuzz.ratio(i, _color_) for i in fash_data["has_color"]}
            max_key1 = max(ratios, key=ratios.get)
        elif d.get("age") and item in d.get("age"):
            _age_ = " ".join(d.get("age"))
            ratios = {i: fuzz.ratio(i, _age_) for i in fash_data["has_age"]}
            max_key2 = max(ratios, key=ratios.get)
        elif d.get("articletype") and item in d.get("articletype"):
            _artType_ = " ".join(d.get("articletype"))
            ratios = {i: fuzz.ratio(i, _artType_) for i in fash_data["has_articleType"]}
            max_key3 = max(ratios, key=ratios.get)
        elif d.get("gender") and item in d.get("gender"):
            _gender_ = " ".join(d.get("gender"))
            ratios = {i: fuzz.ratio(i, _gender_) for i in fash_data["has_gender"]}
            max_key4 = max(ratios, key=ratios.get)
        elif d.get("brandname") and item in d.get("brandname"):
            _brandName_ = " ".join(d.get("brandname"))
            ratios = {i: fuzz.ratio(i, _brandName_) for i in fash_data["has_brandName"]}
            max_key5 = max(ratios, key=ratios.get)
        elif d.get("season") and item in d.get("season"):
            _season_ = " ".join(d.get("season"))
            ratios = {i: fuzz.ratio(i, _season_) for i in fash_data["has_season"]}
            max_key6 = max(ratios, key=ratios.get)
        elif d.get("usage") and item in d.get("usage"):
            _usage_ = " ".join(d.get("usage"))
            ratios = {i: fuzz.ratio(i, _usage_) for i in fash_data["has_usage"]}
            max_key7 = max(ratios, key=ratios.get)

    if _color_ != "":
        try:
            string = string.replace(_color_, " " + max_key1)
        except:
            logger.error("No matches for color")

    if _age_ != "":
        try:
            string = string.replace(_age_, " " + max_key2)
        except:
            logger.error("No matches for age")

    if _artType_ != "":
        try:
            string = string.replace(_artType_, " " + max_key3)
        except:
            logger.error("No matches for article type")

    if _gender_ != "":
        try:
            string = string.replace(_gender_, " " + max_key4)
        except:
            logger.error("No matches for gender")

    if _brandName_ != "":
        try:
            #             print("string",string)
            string = string.replace(_brandName_, " " + max_key5)
        except:
            logger.error("No matches for brand name")

    if _season_ != "":
        try:
            string = string.replace(_season_, " " + max_key6)
        except:
            logger.error("No matches for season")

    if _usage_ != "":
        try:
            string = string.replace(_usage_, " " + max_key7)
        except:
            logger.error("No matches for usage")
    #end = time.time()
    #     print(f"Total runtime of the program in process_query_v3 is {end - begin}")
    return string


def preprocess_entities_relations(entity_dict, relation_dict, entities, relations):
    e = {}
    r = {}

    f = open(entity_dict, "r")
    for line in f:
        line = line.strip().split("\t")
        ent_id = int(line[0])
        ent_name = line[1]
        e[ent_name] = entities[ent_id]
    f.close()

    f = open(relation_dict, "r")
    for line in f:
        line = line.strip().split("\t")
        rel_id = int(line[0])
        rel_name = line[1]
        r[rel_name] = relations[rel_id]
    f.close()
    return e, r


def prepare_embeddings(embedding_dict):
    entity2idx = {}
    idx2entity = {}
    i = 0
    embedding_matrix = []
    for key, entity in embedding_dict.items():
        entity2idx[key.strip()] = i
        idx2entity[i] = key.strip()
        i += 1
        embedding_matrix.append(entity)
    return entity2idx, idx2entity, embedding_matrix


def process_text_file(text_file, split=False):
    data_file = open(text_file, "r")
    data_array = []
    for data_line in data_file.readlines():
        data_line = data_line.strip()
        #         if data_line == '':
        #             continue
        if (data_line == "") or (~("[" in data_line) == -1):
            # print(">>>>>>> IFF Data Line ",data_line)
            continue

        data_line = data_line.strip().split("\t")
        #         print(">>>>>>>>>>> data_line ",data_line)
        question = data_line[0].split("[")
        question_1 = question[0]
        question_2 = question[1].split("]")
        head = question_2[0].strip()
        question_2 = question_2[1]
        question = question_1.strip() + " NE " + question_2.strip()
        ans = data_line[1].split("|")
        data_array.append([head, question.strip(), ans])
    if split == False:
        return data_array


def get_vocab(data):
    word_to_ix = {}
    maxLength = 0
    idx2word = {}
    for d in data:
        sent = d[1]
        for word in sent.split():
            if word not in word_to_ix:
                idx2word[len(word_to_ix)] = word
                word_to_ix[word] = len(word_to_ix)

        length = len(sent.split())
        if length > maxLength:
            maxLength = length
    return word_to_ix, idx2word, maxLength


def data_generator(data, word2ix, entity2idx):
    begin = time.time()

    for i in range(len(data)):
        data_sample = data[i]
        head = entity2idx[data_sample[0].strip()]
        question = data_sample[1].strip()
        #         time.sleep(1)
        end = time.time()
        #         print(f"Total runtime of the program in data_generator is {end - begin}")
        yield torch.tensor(head, dtype=torch.long), question


class KGProdSearch:
    def __init__(
        self,
        entity_path,
        relation_path,
        entity_dict,
        relation_dict,
        w_matrix,
        data_path,
        embedding_folder,
        model_chkpt_file,
    ):
        self.entity_path = entity_path
        self.relation_path = relation_path
        self.entity_dict = entity_dict
        self.relation_dict = relation_dict
        self.w_matrix = w_matrix
        self.data_path = data_path

        self.entities = np.load(self.entity_path)
        self.relations = np.load(self.relation_path)

        self.e, self.r = preprocess_entities_relations(
            self.entity_dict, self.relation_dict, self.entities, self.relations
        )
        self.entity2idx, self.idx2entity, self.embedding_matrix = prepare_embeddings(
            self.e
        )
        self.data = process_text_file(self.data_path, split=False)
        # data = pickle.load(open(data_path, 'rb'))
        self.word2ix, self.idx2word, self.max_len = get_vocab(self.data)
        self.num_hops = 1
        hops = str(self.num_hops)
        self.device = device
        #         print("cuda") if torch.cuda.is_available() else print("cpu")
        self.bn_list = []
        self.embedding_folder = embedding_folder  # "/data/Ashis_Samal/Experiments/Graph_Networks/Data/Fashion_Data/Fashion_KGE_Data/embedding_model/ComplEx"
        # embedding_folder = "/data/Ashis_Samal/Experiments/Graph_Networks/Data/EmbedKGQA/pretrained_models/embeddings/ComplEx_MetaQA_half"
        for i in range(3):
            bn = np.load(
                self.embedding_folder + "/bn" + str(i) + ".npy", allow_pickle=True
            )
            self.bn_list.append(bn.item())

        self.embedding_dim = 256
        self.hidden_dim = 256
        self.relation_dim = 300
        self.freeze = 0
        self.entdrop = 0.1
        self.reldrop = 0.2
        self.scoredrop = 0.2
        self.l3_reg = 0.0
        self.model_name = "ComplEx"
        self.ls = 0.0
        self.model = RelationExtractor(
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            vocab_size=len(self.word2ix),
            num_entities=len(self.idx2entity),
            relation_dim=self.relation_dim,
            pretrained_embeddings=self.embedding_matrix,
            freeze=self.freeze,
            device=self.device,
            entdrop=self.entdrop,
            reldrop=self.reldrop,
            scoredrop=self.scoredrop,
            l3_reg=self.l3_reg,
            model=self.model_name,
            ls=self.ls,
            w_matrix=self.w_matrix,
            bn_list=self.bn_list,
        )
        self.model_chkpt_file = model_chkpt_file
        self.model.load_state_dict(
            torch.load(self.model_chkpt_file, map_location=lambda storage, loc: storage)
        )
        self.model.to(self.device)
        self.model.eval()
        ## sparql fashion meta-data information
        f = open("./fashDict.json", "r")
        self.fash_data = jsonn.load(f)

    def get_products(self, query, top_k):
        begin = time.time()

        # self.model.eval()
        #         print(">>>>>>>>> Input QUery ",query )
        pre = PredictText(query)
        di = {}
        for i in range(len(pre[1][0])):
            di[pre[0][0][i]] = pre[1][0][i]
        data_ = []
        s3, s2 = "", ""
        for i in di.keys():
            if di[i] == "articletype":
                s2 += " " + i + " "
            else:
                s3 += " " + i
        data_ = [[s2.strip(), s3.strip()]]

        #         print("input_data_",data_)
        answers = []
        data_gen = data_generator(
            data=data_, word2ix=self.word2ix, entity2idx=self.entity2idx
        )
        # all_pred_prod_ids = []
        pred_prod_ids = []
        for i in tqdm(range(len(data_))):
            try:
                d = next(data_gen)
                head = d[0].to(self.device)

                question = d[1]

                scores = self.model.get_score_ranked(head=head, sentence=question)[0]

                mask = torch.zeros(len(self.entity2idx)).to(self.device)

                mask[head] = 1
                # reduce scores of all non-candidates
                new_scores = scores - (mask * 99999)

                values, indices = new_scores.topk(top_k)
                #                 print(">>>>>>>  NEW SCORE Result ", values,indices)
                values = values.detach().cpu().numpy()
                values = values[values > 0.1]
                #                 print(">>>>> Values ", values, type(values), values.shape)
                indices = indices.detach().cpu().numpy()
                indices = indices[: values.shape[0]]
                #                 print(">>>>> indices ", indices)
                pred_ans = torch.argmax(new_scores).item()
                #                 print(">>>>>>>>>>>>>>>> pred_ans ",pred_ans)
                for index in indices:
                    pred_prod_ids.append(int(self.idx2entity[index]))
                # print("prod_ids_from_emb",pred_prod_ids)
            #                 print(">>>>>> pred_prod_ids ",pred_prod_ids)
            # all_pred_prod_ids.append(pred_prod_ids)

            except Exception as e:
                #                 print("e",e)
                logger.error("No results from embedding")
        end = time.time()
        #         print(f"Total runtime of the program in get_products is {end - begin}")
        return pred_prod_ids

# get product ids using graph based search 
    def get_prods_from_sparql(self, question, top_k):
        #         begin = time.time()
        query = []
        res = []
        question = question.lower().strip()
        art = ""
        predict = PredictText(question)
        count = 0
        s2, s3, s4, s5, s6, s7, s8 = "", "", "", "", "", "", ""
        for z in range(len(predict[1][0])):
            k = predict[0][0][z]
            i = predict[1][0][z]
            if question != "":
                if i == "articletype":
                    s2 += " " + k + " "
                    count += 1
                    s2 = s2.strip()
                    s2 = s2.replace(" ", "-")
                    question = question.replace(s2, "")
                if i == "brandname":
                    s3 += " " + k + " "
                    s3 = s3.strip()
                    s3 = s3.replace(" ", "-")
                    question = question.replace(s3, "")
                if i == "age":
                    s4 += " " + k + " "
                    s4 = s4.strip()
                    s4 = s4.replace(" ", "-")
                    question = question.replace(s4, "")
                if i == "gender":
                    s5 += " " + k + " "
                    s5 = s5.strip()
                    s5 = s5.replace(" ", "-")
                    question = question.replace(s5, "")
                if i == "color":
                    s6 += " " + k + " "
                    s6 = s6.strip()
                    s6 = s6.replace(" ", "-")
                    question = question.replace(s6, "")
                if i == "usage":
                    s7 += " " + k + " "
                    s7 = s7.strip()
                    s7 = s7.replace(" ", "-")
                    question = question.replace(s7, "")
                if i == "season":
                    s8 += " " + k + " "
                    s8 = s8.strip()
                    s8 = s8.replace(" ", "-")
                    question = question.replace(s8, "")

        if s2:
            art += "?end hp:has_articleType hp:" + s2 + " ."
        if s3:
            art += "?end hp:has_brandName hp:" + s3 + " ."
        if s4:
            art += "?end hp:has_age hp:" + s4 + " ."
        if s5:
            art += "?end hp:has_gender hp:" + s5 + " ."
        if s6:
            art += "?end hp:has_color hp:" + s6 + " ."
        if s7:
            art += "?end hp:has_usage hp:" + s7 + " ."
        if s8:
            art += "?end hp:has_season hp:" + s8 + " ."
        if count >= 1:
            query.append("SELECT ?end WHERE {" + art + "}")
        query = [question.replace("&", "and") for question in query]
        if len(query) > 0:
            sparql_conn.setQuery(
                """
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> 
                PREFIX owl: <http://www.w3.org/2002/07/owl#> 
                PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> 
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> 
                PREFIX dc: <http://purl.org/dc/elements/1.1/> 
                PREFIX hp: <https://deepset.ai/entity/>
                """
                + str(query[0])
            )
            sparql_conn.method = "GET"
            sparql_conn.setReturnFormat(JSON)
            try:
                results = sparql_conn.query().convert()
                for result in results["results"]["bindings"]:
                    try:
                        if result["end"]["value"].split("/")[-1] != None:
                            res.append(int((result["end"]["value"].split("/")[-1])))
                    except:
                        logger.error("No results from SparQL")
            except:
                logger.error("No results from SparQL")

            if len(res) >= top_k:
                #                 print("**from sparql**",res[:top_k])
                #                 time.sleep(1)
                #                 end = time.time()
                #                 print(f"Total runtime of the program in get_prods_from_sparql is {end - begin}")
                return res[:top_k]
            else:
                #                 print("**also from sparql**",res)
                #                 time.sleep(1)
                #                 end = time.time()
                #                 print(f"Total runtime of the program in get_prods_from_sparql is {end - begin}")
                return res
        else:
            #             print("**no results from sparql**")
            #             end = time.time()
            #             print(f"Total runtime of the program in get_prods_from_sparql is {end - begin}")
            return []


entity_path = "./ComplEx/E.npy"
relation_path = "./ComplEx/R.npy"
entity_dict = "./ComplEx/entities.dict"
relation_dict = "./ComplEx/relations.dict"
w_matrix = "./ComplEx/W.npy"
data_path = "./fashion_prd_QA_train_v8.txt"
embedding_folder = "./ComplEx"
model_chkpt_file = "./ComplEx_1__half_best_score_model_V3.chkpt"

obj = KGProdSearch(
    entity_path,
    relation_path,
    entity_dict,
    relation_dict,
    w_matrix,
    data_path,
    embedding_folder,
    model_chkpt_file,
)


@app.route("/kgqa", methods=["POST"])
def kgqa(request):
    #     begin = time.time()
    # print(">>>>> Request ", request.json)
    query = request.json["query"]
    top_k = request.json["top_k"]
    lang = request.json["lang"]
    back_end = str(request.json["backend"])
    ip, routable = get_client_ip(request)
    logger.info({"Query_IP": ip})
    logger.info(
        {"input_query": query, "top_k": top_k, "back_end": back_end, "lang": lang}
    )
    # print(" Request ", request.json)
    # print(query,lang)
    if lang != "en":
        # print("hi")
        query = translate_query(query, lang)
        # print("translate_query",query)
        logger.info({"TranslatedQuery": query})

    query = query.strip().lower()
    if "t shirts" or "t-shirts" in query:
        query = query.replace("t shirts", "tshirts")
        query = query.replace("t-shirts", "tshirts")
    if "gini and jony" or "giniandjony" in query:
        query = query.replace("gini and jony", "gini & jony")
        query = query.replace("giniandjony", "gini&jony")
    #     print(">>>>>>>>raw query>>>>>>",query)
    query = process_query_v3(query)
    logger.info({"process_query_v3": query})
    # print(">>>>>>>>query from process_query_v3 >>>>>>",query)
    #     print(">>>>>> Backend " , back_end)
    if back_end == "sparql":
        pid_list = obj.get_prods_from_sparql(query, top_k)
        # print("sparql_pid_list",pid_list)
        logger.info({"sparql_pid_list": pid_list})
    elif back_end == "embedding":
        pid_list = obj.get_products(query, top_k)
        logger.info({"emb_pid_list": pid_list})
        # print("emb_pid_list",pid_list)

    else:
        #         print(">>>>>> else Backend " , back_end)
        embed_pid_list = obj.get_products(query, top_k)
        #         print(">>>>>> embed_pid_list " , embed_pid_list)
        logger.info({"Hybrid emb_pid_list": embed_pid_list})
        #         print(">>>>>>> embed_pid_list ",embed_pid_list)
        sparql_pid_list = obj.get_prods_from_sparql(query, top_k)
        #         print(">>>>>> sparql_pid_list " , sparql_pid_list)
        logger.info({"Hybrid sparql_pid_list": sparql_pid_list})
        #         print(">>>>>>> sparql_pid_list ",sparql_pid_list)
        pid_list = set(embed_pid_list + sparql_pid_list)
        # print(">>>>>>> ALL pid_list ",pid_list)
        logger.info({"Final hybrid_pid_list": pid_list})

    prod_info = get_prod_info(pid_list)
    #     end = time.time()
    #     print(f"Total runtime of the program in kgqa is {end - begin}")
    return json(prod_info)


@app.route("/kgcategory", methods=["POST"])
def kgqa_category(request):
    return json(category_dict)


@app.route("/language_support", methods=["POST"])
def language_support(request):
    return json(lang_dict)


@app.route("/kgfeedback", methods=["POST"])
def kgqa_feedback(request):
    feedback = request.json["feedback"]
    logger.info({"feedback": feedback})
    return json({"response": "Successfully saved feedback"})


if __name__ == "__main__":
    app.run(port=8081, host="0.0.0.0", access_log=False)

"""
if __name__ == "__main__":
    
    entity_path =  '/home/reverie/data/KGQA_Deploy/ComplEx/E.npy'
    relation_path =  '/home/reverie/data/KGQA_Deploy/ComplEx/R.npy'
    entity_dict =  '/home/reverie/data/KGQA_Deploy/ComplEx/entities.dict'
    relation_dict =  '/home/reverie/data/KGQA_Deploy/ComplEx/relations.dict'
    w_matrix =  '/home/reverie/data/KGQA_Deploy/ComplEx/W.npy'
    data_path = '/home/reverie/data/KGQA_Deploy/fashion_prd_QA_train_v5.txt'
    embedding_folder =  "/home/reverie/data/KGQA_Deploy/ComplEx"
    model_chkpt_file = "/home/reverie/data/KGQA_Deploy/ComplEx_1__half_best_score_model.chkpt"
    obj = KGProdSearch(entity_path,relation_path,entity_dict,relation_dict,w_matrix,data_path,embedding_folder,model_chkpt_file)
    
    #query = "show me Men Black [Tshirts]	51031"
    query = "show me blue levis trouser "
    top_k = 5
    #print(obj.get_products(query,top_k ))
    pid_list = obj.get_products(query,top_k )
    prod_info = get_prod_info(pid_list)
    print(prod_info)
"""
