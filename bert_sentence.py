import torch
import numpy as np
from transformers import BertModel, BertTokenizer
model = BertModel.from_pretrained('./prev_trained_model/bert-base-chinese/')
tokenizer = BertTokenizer.from_pretrained('./prev_trained_model/bert-base-chinese/')

text='''
实验室检验，也称为临床实验室检验或医学实验室检验，是指通过对体液、组织或其他生物样本进行的一系列分析，以获取有关患者健康状况的信息。这些检验有助于疾病的诊断、治疗监控、疾病预防、健康状况评估和医学研究。实验室检验的范围非常广泛，包括但不限于以下几个方面：血液检验：检查血液样本中的红细胞、白细胞、血小板数量，以及血液的成分和生理功能，如血型、血糖、胆固醇、电解质等。尿液检验：分析尿液中的颜色、浓度、pH值、蛋白质、糖、红细胞、白细胞、细菌等，以评估肾脏功能和检测泌尿系统的疾病。生化检验：测定血液或尿液中的各种生化指标，如肝功能、肾功能、血糖、血脂、电解质、激素水平等。微生物检验：通过对体液、组织或其他样本的培养和鉴定，检测和识别病原微生物，如细菌、病毒、真菌和寄生虫。细胞学和病理学检：通过显微镜观察细胞形态和组织的结构，以诊断癌症、炎症、感染等疾病。分子检验：利用分子生物学技术，如PCR（聚合酶链反应），检测DNA、RNA或蛋白质，用于诊断遗传性疾病、感染性疾病和肿瘤等。免疫学检验：检测体内的免疫反应和特定抗体或抗原的存在，用于诊断自身免疫性疾病、过敏反应和疫苗接种后的免疫状态。实验室检验通常由专业的医疗技术人员在临床实验室进行，他们使用各种仪器和试剂来分析样本。检验结果由医生或临床病理学家解释，并与患者的临床症状和其他检查结果一起用于医疗决策。实验室检验是现代医疗保健的重要组成部分，对于疾病的诊断、治疗和管理至关重要
'''
text=text.strip()[:512]
print(len(text))
tokens = tokenizer.tokenize(text)
input_ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = torch.tensor(input_ids).unsqueeze(0)
with torch.no_grad():
        ee=model.get_input_embeddings()
        #word_ee=ee.word_embeddings
        print('ee',ee(input_ids).squeeze(0).shape)
        print(ee(input_ids))
        outputs = model(input_ids)
         
        embeddings = outputs.last_hidden_state[0]
        print(embeddings.shape)
        print(embeddings)
        av=torch.mean(embeddings,0)
        print(av.shape)
        #torch.save(av, 'tes.pt')
