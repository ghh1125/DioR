import numpy as np
import logging
import spacy
import torch
from typing import Any, Dict
from Hullucinations_collector import IntegratedGradientsAnalyzer
from math import exp
from document_RAG import DocumentEvaluator
from scipy.special import softmax
from retriever import BM25, SGPT, SBERT
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from New_keyRAG import KeywordExtractor
import warnings
from TextChunk import TextChunker
import random
import torch.nn.functional as F
from captum.attr import IntegratedGradients
import re
# 将所有警告类别设置为忽略
warnings.filterwarnings('ignore')


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")


# 定义幻觉检测器MLP模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

class BasicGenerator:
    def __init__(self, model_name_or_path):
        logger.info(f"Loading model from {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model_config = AutoConfig.from_pretrained(model_name_or_path,
                                                       trust_remote_code="falcon" in model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto",
                                                          trust_remote_code="falcon" in model_name_or_path)
        self.RNNModel = self.RNN_model(model_path="/home/disk2/ghh/llm-hallucinations/rnn_hallucination_model_0.pth", dropout=0.25)
        if self.model_config.model_type == "llama":
            self.space_token = "▁"
        else:
            self.space_token = self.tokenizer.tokenize(' ')[0]

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, input_text, max_length, return_logprobs=False):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        input_ids = input_ids.to(self.model.device)
        input_length = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)

        if return_logprobs:
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_length,
                return_dict_in_generate=True,
                output_scores=True,
            )
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )

            generated_tokens = outputs.sequences[:, input_length:]
            text = self.tokenizer.decode(generated_tokens[0])  # text = "".join(tokens)
            tokens = [self.tokenizer.decode(t) for t in generated_tokens[0]]
            logprobs = transition_scores[0]
            logprobs = [p.cpu().numpy() for p in logprobs]
            assert len(tokens) == len(logprobs)
            return text, tokens, logprobs

        else:
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_length,
                attention_mask=attention_mask,
            )
            generated_tokens = outputs[:, input_length:]
            text = self.tokenizer.decode(generated_tokens[0])
            return text, None, None

    def RNN_model(self, model_path, dropout, device='cuda'):
        class RNNHallucinationClassifier(torch.nn.Module):
            def __init__(self, dropout=dropout):
                super().__init__()
                hidden_dim = 128
                num_layers = 4
                self.gru = torch.nn.GRU(1, hidden_dim, num_layers, dropout=dropout, batch_first=True,
                                        bidirectional=False)
                self.linear = torch.nn.Linear(hidden_dim, 2)

            def forward(self, seq):
                gru_out, _ = self.gru(seq)
                return self.linear(gru_out)[-1, -1, :]

        model = RNNHallucinationClassifier(dropout=dropout)
        model.load_state_dict(torch.load(model_path))
        model.to(device)  # Ensure the model is moved to the correct device (e.g., GPU)
        model.eval()
        return model

    # def generate_attn_before(self, input_text, max_length, solver="max", use_entropy=False, use_logprob=False,device='cuda'):
    #     print("Input Text:", input_text)
    #
    #     input_ids_num = self.tokenizer(input_text, return_tensors='pt')['input_ids']
    #     input_ids_num = input_ids_num.to(device)  # Ensure the input tensor is on the correct device
    #     input_ids_num = input_ids_num.view(1, -1, 1).to(torch.float)
    #
    #     with torch.no_grad():
    #         output_num = self.RNNModel(input_ids_num)
    #         softmax_result = F.softmax(output_num, dim=-1)
    #         print(f"softmax_result: {softmax_result}")
    #         prediction = torch.argmax(softmax_result, dim=-1)
    #         print(f"Final Prediction: {prediction}")
    #     if prediction == 1:
    #         return 1
    #     else:
    #         return 0

    def generate_attn_before(self, input_text, max_length, solver="max", use_entropy=False, use_logprob=False, device='cuda'):
        # print("Input Text:", input_text)
        questions = re.findall(r"Question:\s*(.*?)(?=\n|Answer:)", input_text, re.DOTALL)
        last_question = questions[-1]
        model_dir = "/home/disk2/ghh/llama"
        analyzer = IntegratedGradientsAnalyzer(model_dir=model_dir)
        attributes_first = analyzer.analyze_text(last_question)

        # print("Integrated Gradients Attributes for the input text:", attributes_first)

        # input_ids_num = self.tokenizer(input_text, return_tensors='pt')['input_ids']
        # input_ids_num = input_ids_num.to(device)  # Ensure the input tensor is on the correct device
        #
        # print(f"Input IDs shape: {input_ids_num.shape}")

        with torch.no_grad():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            attributes_first_tensor = torch.tensor(attributes_first).view(1, -1, 1).to(torch.float).to(device)
            self.RNNModel = self.RNNModel.to(device)
            preds = self.RNNModel(attributes_first_tensor)
            # print(preds.shape)
            preds = torch.nn.functional.softmax(preds, dim=0)
            # print(preds)
            prediction_classes = torch.argmax(preds)
            # print("prediction_classes = ", prediction_classes.item())
        if prediction_classes.item() == 1:
            return 1
        else:
            return 0

    def normalize_attributes(self, attributes: torch.Tensor) -> np.ndarray:
        # Normalize the contribution scores using L2 norm (for each token embedding)
        attributes = attributes.squeeze(0)  # Remove batch dimension
        norm = torch.norm(attributes, dim=1)  # L2 norm along the embedding dimension
        normalized_attributes = norm / torch.sum(norm)  # Normalize the contributions so they add up to 1
        return normalized_attributes.cpu().numpy()  # Convert to numpy for easier handling

    def get_next_token(self, input_ids):
        with torch.no_grad():
            logits = self.model(input_ids).logits
            return logits.squeeze()[-1].argmax()

    def generate_attn(self, input_text, max_length, solver="max", use_entropy=False, use_logprob=False):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        input_ids = input_ids.to(self.model.device)
        input_length = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_length,
            return_dict_in_generate=True,
            output_scores=True,
        )
        generated_tokens = outputs.sequences[:, input_length:]
        tokens = self.tokenizer.convert_ids_to_tokens(generated_tokens[0])
        text = self.tokenizer.decode(generated_tokens[0])

        # merge tokens
        range_ = []
        for i, t in enumerate(tokens):
            # if i == newBM25 or t.startswith(self.space_token) or generated_tokens[newBM25][i] == 13 or tokens[i - newBM25] == '</s>':
                range_.append([i, i])
            # else:
            #     range_[-newBM25][-newBM25] += newBM25

        # attention
        atten = self.model(generated_tokens, output_attentions=True).attentions[-1][0]
        if solver == "max":
            mean_atten, _ = torch.max(atten, dim=1)
            mean_atten = torch.mean(mean_atten, dim=0)
        elif solver == "avg":
            mean_atten = torch.sum(atten, dim=1)
            mean_atten = torch.mean(mean_atten, dim=0)
            for i in range(mean_atten.shape[0]):
                mean_atten[i] /= (mean_atten.shape[0] - i)
        elif solver == "last_token":
            mean_atten = torch.mean(atten[:, -1], dim=0)
        else:
            raise NotImplementedError
        if mean_atten.shape[0] > 1 and tokens[0] == '</s>':
            mean_atten = mean_atten / sum(mean_atten[1:]).item()
        # mean_atten = mean_atten[tl:tr]
        print("loading MLP")
        mlp_model = MLP(2, 64, 1)
        mlp_model.load_state_dict(torch.load('/home/disk_16T/ghh/src/MLP_train/best_mlp_model.pth'))
        mlp_model.eval()
        print("loading MLP dowm")
        seqlist = []
        attns = []
        hallucination_scores = []  # 用于存储每个 token 的幻觉得分

        for r in range_:
            tokenseq = "".join(tokens[r[0]: r[1] + 1]).replace(self.space_token, "")

            # 使用 MLP 模型进行幻觉检测
            input_features = self.tokenizer.encode(tokenseq, return_tensors="pt").float()  # 编码并转换为 float

            # 如果维度不足2，则扩展为2维；如果超过2维，则截取前两维
            if input_features.size(1) < 2:
                pad_size = 2 - input_features.size(1)
                input_features = torch.cat([input_features, torch.zeros(input_features.size(0), pad_size)], dim=1)
            else:
                input_features = input_features[:, :2]

            with torch.no_grad():
                output = mlp_model(input_features)
                hallucination_score = output.item()

            seqlist.append(tokenseq)
            attns.append(mean_atten[r[0]: r[1] + 1].mean().item())
            hallucination_scores.append(hallucination_score)

        # -log prob
        if use_logprob:
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            logprobs = transition_scores[0]
            logprobs = [p.cpu().numpy() for p in logprobs]
            assert len(tokens) == len(logprobs)
            seqlogprobs = []
            for r in range_:
                logprobseq = sum(logprobs[r[0]:r[1] + 1]) / (r[1] - r[0] + 1)
                seqlogprobs.append(logprobseq)
        else:
            seqlogprobs = None

        # entropy
        if use_entropy:
            tmp = []
            for v in outputs.scores:
                tmp.append(v.cpu())
            softmax_probs = softmax(tmp, axis=-1)
            entropies = -np.sum(softmax_probs * np.log(softmax_probs + 1e-10), axis=-1)
            entropies = [v[0] for v in entropies]
            seqentropies = []
            for r in range_:
                entropyseq = sum(entropies[r[0]:r[1] + 1]) / (r[1] - r[0] + 1)
                seqentropies.append(entropyseq)
        else:
            seqentropies = None

        return text, seqlist, attns, seqlogprobs, seqentropies, hallucination_scores


class Counter:
    def __init__(self):
        self.retrieve = 0
        self.generate = 0
        self.hallucinated = 0
        self.token = 0
        self.sentence = 0

    def add_generate(self, text, tokenizer):
        self.generate += 1
        ids = tokenizer(text, return_tensors="pt")['input_ids'][0].tolist()
        self.token += len(ids)
        sentences = [sent.text for sent in nlp(text).sents]
        self.sentence += len(sentences)

    def calc(self, other_counter):
        return {
            "retrieve_count": self.retrieve - other_counter.retrieve,
            "generate_count": self.generate - other_counter.generate,
            "hallucinated_count": self.hallucinated - other_counter.hallucinated,
            "token_count": self.token - other_counter.token,
            "sentence_count": self.sentence - other_counter.sentence
        }


class BasicRAG:
    def __init__(self, args):
        args = args.__dict__
        for k, v in args.items():
            setattr(self, k, v)
        self.generator = BasicGenerator(self.model_name_or_path)
        if "retriever" in self.__dict__:
            self.retriever_type = self.retriever
            if self.retriever_type == "BM25":
                # gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
                self.retriever = BM25(
                    tokenizer=self.generator.tokenizer,
                    index_name="wiki" if "es_index_name" not in args else self.es_index_name,
                    engine="elasticsearch",
                )
            elif self.retriever_type == "SGPT":
                self.retriever = SGPT(
                    model_name_or_path=self.sgpt_model_name_or_path,
                    sgpt_encode_file_path=self.sgpt_encode_file_path,
                    passage_file=self.passage_file
                )
            elif self.retriever_type == "SBERT":
                self.retriever = SBERT(
                    model_name_or_path=self.sbert_model_name_or_path,
                    passage_file=self.passage_file
                )
            else:
                raise NotImplementedError

        self.counter = Counter()

    def retrieve(self, query, topk=1, max_query_length=64):
        self.counter.retrieve += 1
        if self.retriever_type == "BM25":
            _docs_ids, docs = self.retriever.retrieve(
                queries=[query],
                topk=topk,
                max_query_length=max_query_length,
            )
            return docs[0]
        elif self.retriever_type == "SGPT":
            docs = self.retriever.retrieve(
                queries=[query],
                topk=topk,
            )
            if not docs:
                return []
            return docs[0]
        elif self.retriever_type == "SBERT":
            print("OK!")
            query = [query]
            topk = 1
            docs = self.retriever.retrieve(query, topk=1)
            print("docs = ",docs)
            if not docs:
                return []
            return docs[0]
        else:
            raise NotImplementedError

    def get_top_sentence(self, text):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[0] if len(sentences) > 0 else ""

    def get_last_sentence(self, text):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[-1] if len(sentences) > 0 else ""

    def inference(self, question, demo, case):
        # non-retrieval
        assert self.query_formulation == "direct"
        prompt = "".join([d["case"] + "\n" for d in demo])
        prompt += case
        text, _, _ = self.generator.generate(prompt, self.generate_max_length)
        if self.use_counter == True:
            self.counter.add_generate(text, self.generator.tokenizer)
        return text


class SingleRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)

    def inference(self, question, demo, case):
        assert self.query_formulation == "direct"
        docs = self.retrieve(question, topk=self.retrieve_topk)
        # 对 topk 个 passage 生成 prompt
        prompt = "".join([d["case"] + "\n" for d in demo])
        prompt += "Context:\n"
        for i, doc in enumerate(docs):
            prompt += f"[{i + 1}] {doc}\n"
        prompt += "Answer in the same format as before.\n"
        prompt += case
        text, _, _ = self.generator.generate(prompt, self.generate_max_length)
        if self.use_counter == True:
            self.counter.add_generate(text, self.generator.tokenizer)
        return text


class FixLengthRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)

    def inference(self, question, demo, case):
        assert self.query_formulation == "direct"
        text = ""
        while True:
            old_len = len(text)
            docs = self.retrieve(question, topk=self.retrieve_topk)
            # 对 topk 个 passage 生成 prompt
            prompt = "".join([d["case"] + "\n" for d in demo])
            prompt += "Context:\n"
            for i, doc in enumerate(docs):
                prompt += f"[{i + 1}] {doc}\n"
            prompt += "Answer in t he same format as before.\n"
            prompt += case + " " + text
            if self.method == "fix-length-retrieval":
                new_text, _, _ = self.generator.generate(prompt, self.fix_length)
                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                text = text.strip() + " " + new_text.strip()
            else:
                # fix sentence
                new_text, _, _ = self.generator.generate(prompt, self.generate_max_length)
                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                new_text = new_text.strip()
                sentences = list(nlp(new_text).sents)
                sentences = [str(sent).strip() for sent in sentences]
                if len(sentences) == 0:
                    break
                text = text.strip() + " " + str(sentences[0])

            # 判断 token 的个数要少于 generate_max_length
            tokens_count = len(self.generator.tokenizer.encode(text))
            if tokens_count > self.generate_max_length or len(text) <= old_len or "the answer is" in text:
                break
        return text


class TokenRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)

    def modifier(self, text, tokens, logprobs):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]

        tid = 0
        for sid, sent in enumerate(sentences):
            pos = 0
            tr = tid
            while tr < len(tokens):
                apr = sent[pos:].find(tokens[tr])
                if apr == -1:
                    break
                pos = apr + len(tokens[tr])
                tr += 1
            probs = [1 - exp(v) for v in logprobs[tid:tr + 1]]
            probs = np.array(probs)
            p = {
                "avg": np.mean,
                "max": np.max,
                "min": np.min,
            }.get(self.sentence_solver, lambda x: 0)(probs)
            if p > self.hallucination_threshold:  # hallucination
                # keep sentences before hallucination
                prev = "" if sid == 0 else " ".join(sentences[:sid - 1])
                # replace all hallucinated tokens in current sentence with [xxx]
                curr = sentences[sid]
                pos = 0
                # # 这里改成了替换掉最大的那个，而不是所有的
                # max_prob = newBM25
                # for prob, tok in zip(probs, tokens[tid:tr+newBM25]):
                #     max_prob = max(prob, max_prob)
                for prob, tok in zip(probs, tokens[tid:tr + 1]):
                    apr = curr[pos:].find(tok) + pos
                    if prob > self.hallucination_threshold:
                        # if prob == max_prob:
                        curr = curr[:apr] + "[xxx]" + curr[apr + len(tok):]
                        pos = apr + len("[xxx]")
                    else:
                        pos = apr + len(tok)
                return prev, curr, True
            tid = tr + 1

        # No hallucination
        return text, None, False

    def inference(self, question, demo, case):
        # assert self.query_formulation == "direct"
        text = ""
        while True:
            old_len = len(text)
            prompt = "".join([d["case"] + "\n" for d in demo])
            prompt += case + " " + text
            new_text, tokens, logprobs = self.generator.generate(
                prompt,
                self.generate_max_length,
                return_logprobs=True
            )
            if self.use_counter == True:
                self.counter.add_generate(new_text, self.generator.tokenizer)
            ptext, curr, hallucination = self.modifier(new_text, tokens, logprobs)
            if not hallucination:
                text = text.strip() + " " + new_text.strip()
            else:
                if self.query_formulation == "direct":
                    retrieve_question = curr.replace("[xxx]", "")
                elif self.query_formulation == "forward_all":
                    tmp_all = [question, text, ptext]
                    retrieve_question = " ".join(s for s in tmp_all if len(s) > 0)
                else:
                    raise NotImplemented

                docs = self.retrieve(retrieve_question, topk=self.retrieve_topk)
                prompt = "".join([d["case"] + "\n" for d in demo])
                prompt += "Context:\n"
                for i, doc in enumerate(docs):
                    prompt += f"[{i + 1}] {doc}\n"
                prompt += "Answer in the same format as before.\n"
                prompt += case + " " + text + " " + ptext.strip()
                new_text, _, _ = self.generator.generate(prompt, self.generate_max_length)
                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                    self.counter.hallucinated += 1
                text = text.strip() + " " + ptext.strip() + " " + new_text.strip()

            # 判断 token 的个数要少于 generate_max_length
            tokens_count = len(self.generator.tokenizer.encode(text))
            if tokens_count > self.generate_max_length or len(text) <= old_len or "the answer is" in text:
                break
        return text


class EntityRAG(TokenRAG):
    def __init__(self, args):
        super().__init__(args)

    def modifier(self, text, tokens, logprobs):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]

        entity = []
        for sent in sentences:
            doc = nlp(sent)
            li = [ent.text for ent in doc.ents]
            entity.append(li)

        belonging = [-1] * len(text)
        pos = 0
        for tid, tok in enumerate(tokens):
            apr = text[pos:].find(tok) + pos
            assert apr != -1
            for j in range(pos, apr + len(tok)):
                belonging[j] = tid
            pos = apr + len(tok)

        entity_intv = []
        for sid, sent in enumerate(sentences):
            tmp = []
            pos = text.find(sent)
            for ent in entity[sid]:
                apr = text[pos:].find(ent) + pos
                el = belonging[apr]
                er = belonging[apr + len(ent) - 1]
                tmp.append((el, er))
                pos = apr + len(ent)
            entity_intv.append(tmp)

        entity_prob = []
        for ent_itv_per_sent in entity_intv:
            tmp = []
            for itv in ent_itv_per_sent:
                probs = np.array(logprobs[itv[0]:itv[1] + 1])
                p = {
                    "avg": np.mean,
                    "max": np.max,
                    "min": np.min,
                    "first": lambda x: x[0] if len(x) > 0 else 0
                }.get(self.entity_solver, lambda x: 0)(probs)
                tmp.append(p)
            entity_prob.append(tmp)

        for sid in range(len(sentences)):
            if len(entity_prob[sid]) == 0:
                continue
            probs = [1 - exp(v) for v in entity_prob[sid]]
            probs = np.array(probs)
            p = {
                "avg": np.mean,
                "max": np.max,
                "min": np.min,
            }.get(self.sentence_solver, lambda x: 0)(probs)
            if p > self.hallucination_threshold:  # hallucination
                # keep sentences before hallucination
                prev = "" if sid == 0 else " ".join(sentences[:sid - 1])
                # replace all hallucinated entities in current sentence with [xxx]
                curr = sentences[sid]
                pos = 0
                for prob, ent in zip(probs, entity[sid]):
                    apr = curr[pos:].find(ent) + pos
                    if prob > self.hallucination_threshold:
                        curr = curr[:apr] + "[xxx]" + curr[apr + len(ent):]
                        pos = apr + len("[xxx]")
                    else:
                        pos = apr + len(ent)
                return prev, curr, True
        # No hallucination
        return text, None, False

    def inference(self, question, demo, case):
        return super().inference(question, demo, case)


class AttnWeightRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
        self.i = 0
        self.num_docs = 0
        self.count_search = 0

    def modifier(self, text, tokens, attentions, weight):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        tid = 0
        for sid, sent in enumerate(sentences):
            tl, tr = tid, tid
            if sid == len(sentences) - 1:
                tl, tr = tid, len(tokens)
            else:
                for i in range(tid + 1, len(tokens)):
                    seq = " ".join(tokens[tl:i])
                    if sent in seq:
                        tr = i
                        break
                tid = tr
            # value = attenion * (-log prob)
            attns = attentions[tl:tr]
            attns = np.array(attns) / sum(attns)
            value = [attns[i - tl] * weight[i] * (tr - tl) for i in range(tl, tr)]
            thres = [1 if v > self.hallucination_threshold else 0 for v in value]
            if 1 in thres:
                # hallucinated
                if "check_real_words" in self.__dict__ and self.check_real_words:
                    doc = nlp(sent)
                    real_words = set(token.text for token in doc if token.pos_ in
                                     ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])

                    def match(tok):
                        for word in real_words:
                            if word in tok:
                                return True
                        return False

                    for i in range(len(thres)):
                        if not match(tokens[tl + i]):
                            thres[i] = 0

                prev = "" if sid == 0 else " ".join(sentences[:sid - 1])
                # curr = " ".join(
                #     [tokens[i] if thres[i] == newBM25 else "[xxx]" for i in range(len(thres))]
                # )
                return True, prev, tokens[tl:tr], thres
        return False, text, None, None

    def keep_real_words(self, prev_text, curr_tokens, curr_hit):
        curr_text = " ".join(curr_tokens)
        all_text = prev_text + " " + curr_text
        input_ids = self.generator.tokenizer.encode(all_text, return_tensors="pt")
        input_length = input_ids.shape[1]
        tokens_tmp = self.generator.tokenizer.convert_ids_to_tokens(input_ids[0])

        atten_tmp = self.generator.model(input_ids, output_attentions=True).attentions[-1][0]

        # merge tokens
        range_ = []
        for i, t in enumerate(tokens_tmp):
            if i == 0 or t.startswith(self.generator.space_token) or input_ids[0][i] == 13:
                range_.append([i, i])
            else:
                range_[-1][-1] += 1
        tokens = []
        for r in range_:
            tokenseq = "".join(tokens_tmp[r[0]: r[1] + 1]).replace(self.generator.space_token, "")
            tokens.append(tokenseq)

        # 获取幻觉词对应的 attention
        curr_st = len(tokens) - len(curr_tokens)
        atten_tmp = torch.mean(atten_tmp, dim=0)
        attns = []
        for r in range_:
            # att = torch.zeros(atten_tmp.shape[newBM25], input_length)
            att = torch.zeros(input_length)
            for i in range(r[0], r[1] + 1):
                if i == 0:
                    continue
                v = atten_tmp[i - 1][:r[0]]  # 上一位的
                v = v / v.sum()
                t = torch.zeros(input_length)
                t[:r[0]] = v
                att += t
            att /= (r[1] - r[0] + 1)
            # merge token for att
            att = torch.tensor([att[rr[0]:rr[1] + 1].sum() for rr in range_])
            attns.append(att)

        # 计算每个超过阈值的 token 在前文的 attentions
        forward_attns = torch.zeros(len(tokens))
        hit_cnt = 0
        for i in range(len(curr_hit)):
            if curr_hit[i] == 1:
                forward_attns += attns[curr_st + i]
                hit_cnt += 1
        forward_attns /= hit_cnt
        forward_attns = forward_attns.tolist()

        # Calculate word embeddings and their cosine similarity
        def cosine_similarity(vec1, vec2):
            vec1 = np.array(vec1).flatten()
            vec2 = np.array(vec2).flatten()
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

        # 计算TF-IDF分数
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_scores = tfidf_vectorizer.fit_transform([all_text]).toarray().flatten()

        # 分析词性，保留实词对应的 attns
        doc = nlp(all_text)
        real_words = set(token.text for token in doc if token.pos_ in ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])

        def match(token):
            for word in real_words:
                if word in token:
                    return True
            return False

        scaler = MinMaxScaler()

        word_embeddings = {token.text: token.vector for token in doc}  # 获取每个 token 的词向量
        mean_vector = np.mean(list(word_embeddings.values()), axis=0)  # 计算平均词向量

        real_pairs = []

        # 标准化注意力权重
        att_array = np.array(forward_attns).reshape(-1, 1)
        att_array = np.nan_to_num(att_array)  # 处理 NaN 值
        att_normalized = scaler.fit_transform(att_array).flatten()

        # 标准化 TF-IDF 得分
        tfidf_array = np.array(tfidf_scores).reshape(-1, 1)
        tfidf_array = np.nan_to_num(tfidf_array)  # 处理 NaN 值
        tfidf_normalized = scaler.fit_transform(tfidf_array).flatten()

        for i in range(len(tokens)):
            tok = tokens[i]
            att = att_normalized[i]
            tfidf_score = tfidf_normalized[i] if i < len(tfidf_normalized) else 0

            if i >= curr_st and curr_hit[i - curr_st]:
                continue

            if match(tok):
                embeddings = word_embeddings.get(tok, mean_vector)
                similarities = [cosine_similarity(embeddings, word_embeddings.get(other_tok, mean_vector)) for other_tok
                                in tokens]
                mean_similarity = sum(similarities) / len(similarities) if similarities else 0.0

                # 处理 NaN 值
                mean_similarity = 0.0 if np.isnan(mean_similarity) else mean_similarity

                # 标准化相似度得分
                similarity_array = np.array([mean_similarity]).reshape(-1, 1)
                similarity_array = np.nan_to_num(similarity_array)  # 处理 NaN 值
                similarity_normalized = scaler.fit_transform(similarity_array).flatten()[0]

                combined_score = att + tfidf_score + similarity_normalized
                real_pairs.append((combined_score, tok, i))

        if "retrieve_keep_top_k" in self.__dict__:
            top_k = min(self.retrieve_keep_top_k, len(real_pairs))
        elif "retrieve_keep_ratio" in self.__dict__:
            top_k = int(len(real_pairs) * self.retrieve_keep_ratio)

        real_pairs = sorted(real_pairs, key=lambda x: x[0], reverse=True)
        real_pairs = real_pairs[:top_k]
        real_pairs = sorted(real_pairs, key=lambda x: x[2])
        return " ".join([x[1] for x in real_pairs])

    def inference(self, question, demo, case):
        # assert self.query_formulation == "direct"
        # print(question)
        # print("#" * 20)
        text = ""
        final_text = ""
        self.i = 1
        while True:
            old_len = len(text)
            prompt = "".join([d["case"] + "\n" for d in demo])
            tmp_li = [case, text]
            prompt += " ".join(s for s in tmp_li if len(s) > 0)
            # print('####', prompt)
            # prompt += case + " " + text
            beforeHullucinated = 1
            if self.i == 0:
                logger.info("Prepare Early Detection")
                beforeHullucinated = self.generator.generate_attn_before(prompt,
                    self.generate_max_length,
                    # self.attention_solver,
                    use_entropy=self.method == "ConRek",
                    use_logprob=self.method == "attn_prob"
                )
            if beforeHullucinated == 0 and self.i == 0:
                # print("i = ",self.i)
                # logger.info("Start Early Detection")
                self.i = self.i + 1
                # LLM no Ability to Answer
                # new_text, tokens, attns, logprobs, entropies, hallucinated = self.generator.generate_attn(
                #     prompt,
                #     self.generate_max_length,
                #     # self.attention_solver,
                #     use_entropy=self.method == "ConRek",
                #     use_logprob=self.method == "attn_prob"
                # )
                # weight = entropies if self.method == "ConRek" else [-v for v in logprobs]
                #
                # if self.use_counter == True:
                #     self.counter.add_generate(new_text, self.generator.tokenizer)
                #
                # hallucination, ptext, curr_tokens, curr_hit = self.modifier(new_text, tokens, attns, weight)
                #
                # forward_all = [question, text, ptext]
                # forward_all = " ".join(s for s in forward_all if len(s) > 0)
                #
                # def fetch_last_n_tokens(text, num, tokenizer=self.generator.tokenizer):
                #     tokens = tokenizer.tokenize(text)
                #     if num >= len(tokens):
                #         return text
                #     last_n_tokens = tokens[-num:]
                #     last_n_sentence = ' '.join(last_n_tokens)
                #     return last_n_sentence
                #
                # if self.query_formulation == "current":
                #     retrieve_question = " ".join(curr_tokens)
                #
                # elif self.query_formulation == "current_wo_wrong":
                #     retrieve_question = " ".join(
                #         list(curr_tokens[i] if curr_hit[i] == 0 else "" for i in range(len(curr_tokens)))
                #     )
                #
                # elif self.query_formulation == "forward_all":
                #     retrieve_question = forward_all
                #
                # elif self.query_formulation == "last_sentence":
                #     retrieve_question = self.get_last_sentence(forward_all)
                #
                # elif self.query_formulation == "last_n_tokens":
                #     assert "retrieve_keep_top_k" in self.__dict__
                #     retrieve_question = fetch_last_n_tokens(
                #         forward_all, self.retrieve_keep_top_k)
                #
                # elif self.query_formulation == "real_words":
                #     retrieve_question = self.keep_real_words(
                #         prev_text=prompt,
                #         curr_tokens=curr_tokens,
                #         curr_hit=curr_hit,
                #     )
                # else:
                #     raise NotImplemented

                retrieve_question =  re.findall(r"Question:\s*(.*?)(?=\n|Answer:)", prompt, re.DOTALL)
                last_question = retrieve_question[-1]


                doc = nlp(last_question)

                # 提取名词、动词、形容词等作为关键词
                keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN', 'NUM']]
                keywords = " ".join(keywords)

                topk = self.retrieve_topk

                if len(keywords) < self.retrieve_topk:
                    topk = len(keywords)
                # print("keywords = ", keywords)
                docs = self.retrieve(keywords, topk=topk)
                prompt = "".join([d["case"] + "\n" for d in demo])
                prompt += "Context:\n"
                for i, doc in enumerate(docs):
                    prompt += f"[{i + 1}] {doc}\n"
                prompt += "Answer in the same format as before.\n"
                text = ""
                ptext = ""
                tmp_li = [case, text, ptext.strip()]
                prompt += " ".join(s for s in tmp_li if len(s) > 0)
                # print('#####', prompt)
                # prompt += case + " " + text + " " + ptext.strip()
                new_text, _, _ = self.generator.generate(prompt, self.generate_max_length)
                # print(new_text)
                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                    self.counter.hallucinated += 1
                new_text = self.get_top_sentence(new_text)
                tmp_li = [text.strip(), ptext.strip(), new_text.strip()]
                text = " ".join(s for s in tmp_li if len(s) > 0)
                # print(text)
            else:
                logger.info("Hallucination Detection")
                # print(text)
                new_text, tokens, attns, logprobs, entropies, hallucinated = self.generator.generate_attn(
                    prompt,
                    self.generate_max_length,
                    # self.attention_solver,
                    use_entropy=self.method == "ConRek",
                    use_logprob=self.method == "attn_prob"
                )
                weight = entropies if self.method == "ConRek" else [-v for v in logprobs]

                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer)

                hallucination, ptext, curr_tokens, curr_hit = self.modifier(new_text, tokens, attns, weight)

                mean_score = sum(hallucinated) / len(hallucinated)
                flag = 0
                # 判断每个token是否偏向于幻觉
                for score in hallucinated:
                    if score > mean_score and score <= 0.5:
                        flag = 1

                if not hallucination and flag == 0:
                    text = text.strip() + " " + new_text.strip()
                else:
                    forward_all = [question, text, ptext]
                    forward_all = " ".join(s for s in forward_all if len(s) > 0)

                    def fetch_last_n_tokens(text, num, tokenizer=self.generator.tokenizer):
                        tokens = tokenizer.tokenize(text)
                        if num >= len(tokens):
                            return text
                        last_n_tokens = tokens[-num:]
                        last_n_sentence = ' '.join(last_n_tokens)
                        return last_n_sentence

                    if self.query_formulation == "current":
                        retrieve_question = " ".join(curr_tokens)

                    elif self.query_formulation == "current_wo_wrong":
                        retrieve_question = " ".join(
                            list(curr_tokens[i] if curr_hit[i] == 0 else "" for i in range(len(curr_tokens)))
                        )

                    elif self.query_formulation == "forward_all":
                        retrieve_question = forward_all

                    elif self.query_formulation == "last_sentence":
                        retrieve_question = self.get_last_sentence(forward_all)

                    elif self.query_formulation == "last_n_tokens":
                        assert "retrieve_keep_top_k" in self.__dict__
                        retrieve_question = fetch_last_n_tokens(
                            forward_all, self.retrieve_keep_top_k)

                    elif self.query_formulation == "real_words":
                        retrieve_question = self.keep_real_words(
                            prev_text=question + " " + text + " " + ptext,
                            curr_tokens=curr_tokens,
                            curr_hit=curr_hit,
                        )
                    else:
                        raise NotImplemented

                    # print("retrieve_question = ", retrieve_question)


                    docs = self.retrieve(retrieve_question, topk=self.retrieve_topk)
                    print("docs = ", docs)
                    # 优化文档检索
                    old_docs = docs
                    if len(old_docs) > 0:
                        # logger.info(f"{old_docs} NOT None!")
                        evaluator = DocumentEvaluator(retrieve_question)

                        evaluation_results = evaluator.evaluate_retrieved_documents(docs)
                        relevance_scores = evaluation_results['relevance_scores']
                        mean_score = evaluation_results['avg_relevance_score']
                        final_docs = []
                        self.num_docs = 0
                        self.count_search = 0
                        # print("num_docs1 = ", self.num_docs)
                        # logger.info("RAG Optimization Start")
                        # logger.info(f"Start RAG Optimization with num_docs = {self.num_docs}")
                        while self.num_docs < self.retrieve_topk:
                            # logger.info(f"num_docs = {self.num_docs}, retrieve_topk = {self.retrieve_topk}")
                            for i, doc in enumerate(docs):
                                doc_text = doc
                                score = relevance_scores[i]
                                # print(doc_text)
                                # logger.debug(f"Doc {i} score: {score}, mean_score: {mean_score}")

                                if score >= mean_score:
                                    # logger.info(f"Doc {i} added to final_docs")
                                    # logger.info("Document slicing at the same time")
                                    self.num_docs += 1
                                    text_Chunk = TextChunker()
                                    chunks = text_Chunk.chunk_text(doc_text, retrieve_question)
                                    final_docs.extend(chunks)
                                    if final_text == None:
                                        logger.info("ERROR!!")

                                if self.num_docs == self.retrieve_topk:
                                    # logger.info("RAG Optimization Stop")
                                    break
                            if self.num_docs < self.retrieve_topk:
                                # logger.info("Not enough docs yet, retrieving new documents...")
                                self.count_search = self.count_search + 1
                                if self.count_search == 10:
                                    if self.num_docs < self.retrieve_topk:
                                        new_docs = self.retrieve(retrieve_question,
                                                                 topk=self.retrieve_topk - self.num_docs)
                                        if len(new_docs) > 0:
                                            for l, doc_new in enumerate(new_docs):
                                                new_doc = doc_new
                                                new_Chunk = TextChunker()
                                                new_chunk = new_Chunk.chunk_text(new_doc, retrieve_question)
                                                final_docs.extend(new_chunk)
                                    self.count_search = 0
                                    break
                                keywordExtractor = KeywordExtractor()
                                if final_docs == None:
                                    docs = old_docs
                                    break
                                # print("final_docs =" , final_docs)
                                new_key = keywordExtractor.extract_keywords(final_docs)
                                new_key_str = " ".join(new_key)
                                retrieve_question = retrieve_question + new_key_str
                                # logger.info(f"New new_key_str: {new_key_str}")
                                topk = self.retrieve_topk * 5
                                # logger.info(f"Topk value used for document retrieval: {topk}")
                                new_docs = self.retrieve(retrieve_question, topk=topk)
                                if any(doc in final_docs for doc in new_docs):
                                    # logger.info(
                                    #     "There are duplicate documents in new_docs that already exist in final_docs.")
                                    new_docs = [doc for doc in new_docs if doc not in final_docs]
                                evaluation_results = evaluator.evaluate_retrieved_documents(new_docs)
                                relevance_scores = evaluation_results['relevance_scores']
                                mean_score = evaluation_results['avg_relevance_score']
                                docs = new_docs


                    # while self.num_docs < self.retrieve_topk:
                    #     print("num_docs2 = ", self.num_docs)
                    #     for i, doc in enumerate(docs):
                    #         doc_text = doc
                    #         score = relevance_scores[i]
                    #
                    #         if score > mean_score:
                    #             if self.num_docs == self.retrieve_topk:
                    #                 logger.info("RAG Optimization Stop")
                    #                 break
                    #             self.num_docs = self.num_docs + 1
                    #             final_docs.append(doc_text)
                    #     if self.num_docs < self.retrieve_topk:
                    #         keywordExtractor = KeywordExtractor()
                    #         new_key = keywordExtractor.extract_keywords(final_docs)
                    #         new_key_str = " ".join(new_key)
                    #         retrieve_question = retrieve_question + new_key_str
                    #         # print("retrieve_question = ", retrieve_question)
                    #         new_docs = self.retrieve(retrieve_question, topk=self.retrieve_topk)
                    #         evaluation_results = evaluator.evaluate_retrieved_documents(new_docs)
                    #         relevance_scores = evaluation_results['relevance_scores']
                    #         mean_score = evaluation_results['avg_relevance_score']


                        if final_docs == None:
                            docs = old_docs
                        else:
                            docs = final_docs
                    # print(docs)

                    else:
                        pass

                    prompt = "".join([d["case"] + "\n" for d in demo])
                    prompt += "Context:\n"
                    for i, doc in enumerate(docs):
                        prompt += f"[{i + 1}] {doc}\n"
                    prompt += "Answer in the same format as before.\n"
                    tmp_li = [case, text, ptext.strip()]
                    prompt += " ".join(s for s in tmp_li if len(s) > 0)
                    # print('#####', prompt)
                    # prompt += case + " " + text + " " + ptext.strip()
                    new_text, _, _ = self.generator.generate(prompt, self.generate_max_length)
                    if self.use_counter == True:
                        self.counter.add_generate(new_text, self.generator.tokenizer)
                        self.counter.hallucinated += 1
                    new_text = self.get_top_sentence(new_text)
                    tmp_li = [text.strip(), ptext.strip(), new_text.strip()]
                    text = " ".join(s for s in tmp_li if len(s) > 0)
                    # text = text.strip() + " " + ptext.strip() + " " + new_text.strip()
                    # print(text)
                    # print("### retrieve_question ###")
                    # print(retrieve_question)
                    # context = "### Context: ###\n"
                    # for i, doc in enumerate(docs):
                    #     context += f"[{i+newBM25}] {doc}\n"
                    # print(context)
                    # print(text)

                # 判断 token 的个数要少于 generate_max_length
                tokens_count = len(self.generator.tokenizer.encode(text))
                if tokens_count > self.generate_max_length or len(text) <= old_len or "the answer is" in text:
                    break
        # else:
        #     new_text, tokens, attns, logprobs, entropies, hallucinated= self.generator.generate_attn(
        #         prompt,
        #         self.generate_max_length,
        #         # self.attention_solver,
        #         use_entropy=self.method == "ConRek",
        #         use_logprob=self.method == "attn_prob"
        #     )
        #     weight = entropies if self.method == "ConRek" else [-v for v in logprobs]
        #
        #     if self.use_counter == True:
        #         self.counter.add_generate(new_text, self.generator.tokenizer)
        #
        #
        #     hallucination, ptext, curr_tokens, curr_hit = self.modifier(new_text, tokens, attns, weight)
        #
        #     mean_score = sum(hallucinated) / len(hallucinated)
        #     flag = 0
        #     # 判断每个token是否偏向于幻觉
        #     for score in hallucinated:
        #         if score > mean_score and score <= 0.5:
        #             flag = 1
        #
        #     if not hallucination and flag == 0:
        #         text = text.strip() + " " + new_text.strip()
        #     else:
        #         forward_all = [question, text, ptext]
        #         forward_all = " ".join(s for s in forward_all if len(s) > 0)
        #
        #         def fetch_last_n_tokens(text, num, tokenizer=self.generator.tokenizer):
        #             tokens = tokenizer.tokenize(text)
        #             if num >= len(tokens):
        #                 return text
        #             last_n_tokens = tokens[-num:]
        #             last_n_sentence = ' '.join(last_n_tokens)
        #             return last_n_sentence
        #
        #         if self.query_formulation == "current":
        #             retrieve_question = " ".join(curr_tokens)
        #
        #         elif self.query_formulation == "current_wo_wrong":
        #             retrieve_question = " ".join(
        #                 list(curr_tokens[i] if curr_hit[i] == 0 else "" for i in range(len(curr_tokens)))
        #             )
        #
        #         elif self.query_formulation == "forward_all":
        #             retrieve_question = forward_all
        #
        #         elif self.query_formulation == "last_sentence":
        #             retrieve_question = self.get_last_sentence(forward_all)
        #
        #         elif self.query_formulation == "last_n_tokens":
        #             assert "retrieve_keep_top_k" in self.__dict__
        #             retrieve_question = fetch_last_n_tokens(
        #                 forward_all, self.retrieve_keep_top_k)
        #
        #         elif self.query_formulation == "real_words":
        #             retrieve_question = self.keep_real_words(
        #                 prev_text=question + " " + text + " " + ptext,
        #                 curr_tokens=curr_tokens,
        #                 curr_hit=curr_hit,
        #             )
        #         else:
        #             raise NotImplemented
        #
        #         docs = self.retrieve(retrieve_question, topk=self.retrieve_topk)
        #         prompt = "".join([d["case"] + "\n" for d in demo])
        #         prompt += "Context:\n"
        #         for i, doc in enumerate(docs):
        #             prompt += f"[{i + 1}] {doc}\n"
        #         prompt += "Answer in the same format as before.\n"
        #         tmp_li = [case, text, ptext.strip()]
        #         prompt += " ".join(s for s in tmp_li if len(s) > 0)
        #         # print('#####', prompt)
        #         # prompt += case + " " + text + " " + ptext.strip()
        #         new_text, _, _ = self.generator.generate(prompt, self.generate_max_length)
        #         if self.use_counter == True:
        #             self.counter.add_generate(new_text, self.generator.tokenizer)
        #             self.counter.hallucinated += 1
        #         new_text = self.get_top_sentence(new_text)
        #         tmp_li = [text.strip(), ptext.strip(), new_text.strip()]
        #         text = " ".join(s for s in tmp_li if len(s) > 0)
        #         # text = text.strip() + " " + ptext.strip() + " " + new_text.strip()
        #
        #         # print("### retrieve_question ###")
        #         # print(retrieve_question)
        #         # context = "### Context: ###\n"
        #         # for i, doc in enumerate(docs):
        #         #     context += f"[{i+newBM25}] {doc}\n"
        #         # print(context)
        #         # print(text)
        #
        #     # 判断 token 的个数要少于 generate_max_length
        #     tokens_count = len(self.generator.tokenizer.encode(text))
        #     if tokens_count > self.generate_max_length or len(text) <= old_len or "the answer is" in text:
        #         break
        # print("#" * 20)
        self.i = 0
        return text
