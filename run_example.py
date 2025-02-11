# coding: utf-8
"""
BERT-PCA: 在BERT embedding后面接一个PCA, 得到句子的句向量表示, 
用于计算两段文本相似度.

@env: python3, pytorch>=1.7.1, transformers==4.2.0
@author: Weijie Liu
@date: 20/01/2020
"""
import os
import torch,json
import numpy as np
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import scipy.stats

TRAIN_PATH = './data/downstream/STS/STSBenchmark/sts-train.csv'
DEV_PATH = './data/downstream/STS/STSBenchmark/sts-dev.csv'
TEST_PATH = './data/downstream/STS/STSBenchmark/sts-test.csv'
#MODEL_NAME = './prev_trained_model/medbert-base-chinese/'#bert-base-chinese/'
MODEL_NAME = './prev_trained_model/bert-base-chinese/'
#MODEL_NAME = './model/bert-base-uncased' # 本地模型文件
# MODEL_NAME = './model/bert-large-uncased' # 本地模型文件
# MODEL_NAME = './model/sbert-base-uncased-nli' # 本地模型文件
# MODEL_NAME = './model/sbert-large-uncased-nli' # 本地模型文件

POOLING = 'first_last_avg'
# POOLING = 'last_avg'
# POOLING = 'last2avg'

USE_WHITENING = True
N_COMPONENTS = 768 
MAX_LENGTH = 512 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _read_json_text(input_file):
        lines = []
        with open(input_file,'r') as f:
            for line in f.readlines():
                content = json.loads(line.encode().decode('utf-8-sig'))
                text = content["originalText"]
                lines.append(text)
        print(len(lines))
        return lines
def load_dataset(path, test_or_train):
    """
    loading training or testing dataset.
    """
    senta_batch, sentb_batch, scores_batch = [], [], []
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            items = line.strip().split('\t')
            if test_or_train == 'train':
                senta, sentb, score = items[-2], items[-1], float(items[-3])
            elif test_or_train in ['dev', 'test']:
                senta, sentb, score = items[-2], items[-1], float(items[-3])
            else:
                raise Exception("{} error".format(test_or_train))
            senta_batch.append(senta)
            sentb_batch.append(sentb)
            scores_batch.append(score)
    return senta_batch, sentb_batch, scores_batch


def build_model(name):
    tokenizer = BertTokenizer.from_pretrained(name)
    model = BertModel.from_pretrained(name)
    model = model.to(DEVICE)
    return tokenizer, model


def sents_to_vecs(sents, tokenizer, model):
    vecs = []
    with torch.no_grad():
        for sent in tqdm(sents):
            inputs = tokenizer(sent, return_tensors="pt", padding=True, truncation=True,  max_length=MAX_LENGTH)
            inputs['input_ids'] = inputs['input_ids'].to(DEVICE)
            inputs['token_type_ids'] = inputs['token_type_ids'].to(DEVICE)
            inputs['attention_mask'] = inputs['attention_mask'].to(DEVICE)

            hidden_states = model(**inputs, return_dict=True, output_hidden_states=True).hidden_states

            if POOLING == 'first_last_avg':
                output_hidden_state = (hidden_states[-1] + hidden_states[1]).mean(dim=1)
            elif POOLING == 'last_avg':
                output_hidden_state = (hidden_states[-1]).mean(dim=1)
            elif POOLING == 'last2avg':
                output_hidden_state = (hidden_states[-1] + hidden_states[-2]).mean(dim=1)
            else:
                raise Exception("unknown pooling {}".format(POOLING))

            vec = output_hidden_state.cpu().numpy()[0]
            vecs.append(vec)
    assert len(sents) == len(vecs)
    vecs = np.array(vecs)
    return vecs


def calc_spearmanr_corr(x, y):
    return scipy.stats.spearmanr(x, y).correlation


def compute_kernel_bias(vecs, n_components):
    """计算kernel和bias
    最后的变换：y = (x + bias).dot(kernel)
    """
    vecs = np.concatenate(vecs, axis=0)
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(s**0.5))
    W = np.linalg.inv(W.T)
    W = W[:, :n_components]
    return W, -mu


def transform_and_normalize(vecs, kernel, bias):
    """应用变换，然后标准化
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5


def normalize(vecs):
    """标准化
    """
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5


def main():

    print(f"Configs: {MODEL_NAME}-{POOLING}-{USE_WHITENING}-{N_COMPONENTS}.")
    c1="""
    微生物（microorganisms），通常被称为微生物或微菌，是一类非常小的生物体，它们存在于自然界中的几乎每一个环境中，包括土壤、水、空气、动植物体内以及人类体内。微生物可以是单细胞的，如细菌和原生生物，也可以是多细胞的，如某些真菌和藻类，还有一些没有细胞结构的微生物，如病
    """
    c2="""
    医学程序（medical procedures）是指为了诊断、治疗或预防疾病而进行的各种操作和程序。这些程序可以包括简单的门诊操作，如伤口缝合或疫苗接种，也可以包括复杂的手术，如心脏搭桥手术或器官移植
    """
    c3="""
    临床表现（clinical manifestations）是指患者因疾病而产生的可以被医生观察到或测量的症状和体征。临床表现是医生用来诊断疾病和制定治疗计划的重要信息来源
    """
    c4="""
    医学检查（medical examination），也称为体检或身体检查，是一种通过观察、触摸、听诊、实验室测试和其他诊断程序来评估一个人健康状况的过程。医学检查可以是为了筛查疾病、评估健康状况、监测慢性疾病、诊断特定的症状或疾病
    """
    c5="""
    “equipment” 一词指的是用于诊断、治疗、监测患者健康状况的各种工具和设备。医学设备可以是非常简单的，如体温计和血压计，也可以是非常复杂的，如磁共振成像（MRI）扫描仪和线性加速器（用于放射治疗）
    """
    c6="""
    "department"一词通常指的是医院或医疗机构内部的组织单位，每个部门专注于特定类型的医疗服务或医学专业。医学部门通常由一群专业的医疗人员组成，包括医生、护士、技术人员和其他健康护理专业人员，他们共同协作，为患者提供特定的医疗服
    """
    c7="""
    "body"一词通常指的是人体的物理结构，包括所有的器官、组织、细胞和其他生物组成成分。人体是一个复杂的生物系统，由多个相互作用的系统组成，如呼吸系统、循环系统、消化系统、神经系统、内分泌系统、免疫系统、肌肉骨骼系统等
    """
    c8="""
    dis疾病是指人体在一定条件下，由于内外因素的作用，导致正常生理功能紊乱，出现异常的形态结构、代谢和功能变化，表现为一系列症状和体征的异常生命过程。疾病可以是遗传的、感染的、职业的、环境的、心理的或生活方式相关的等多种原因引起
    """
    c9="""
    Drugs 药物是一种用于预防、诊断、治疗或缓解疾病症状的物质或组合物。药物可以通过不同的方式作用于人体，包括：改变或调节生理功能,抑制或杀灭病原体,缓解症状,替代或补充缺失的物质,影响心理状态和行为。药物可以以多种形式存在，包括：固体：片剂、胶囊、粉末等。液体：溶液、悬浮液、注射液等。半固体：凝胶、乳膏、膏药等
    """

    a="""
    疾病是指人体在一定条件下，由于内外因素的作用，导致正常生理功能紊乱，出现异常的形态结构、代谢和功能变化，表现为一系列症状和体征的异常生命过程。疾病可以是遗传的、感染的、职业的、环境的、心理的或生活方式相关的等多种原因引起。诊断则是医生根据患者的症状、体征、实验室检查、影像学检查等资料，运用医学知识和临床经验，对患者的疾病进行分析、判断和鉴别，从而确定疾病的性质和病因的过程。准确的诊断是制定治疗计划、改善患者预后的基础。在诊断过程中，医生会采取病史采集、体格检查、实验室检测、影像学检查等方法，综合分析各种信息，以得出最可能的诊断。随着科技的发展，医学诊断的方法和工具也在不断进步，如分子诊断、基因检测等，使得诊断更加精确和早期
    """
    b="""
    手术，也称为外科手术，是一种通过物理手段进入人体内部或表面，进行诊断、治疗或改善身体状况的医疗程序。手术通常由经过专业训练的外科医生（外科学家）或牙科外科医生执行，伴随着麻醉师提供的麻醉以减轻疼痛和保证患者的舒适度。
    手术的种类繁多，可以根据多种标准进行分类，包括：手术范围：可分为大手术、中等手术和小手术。手术方式：开放式手术：通过较大的皮肤切口进行，以便医生可以直接观察和操作手术区域。微创手术：通过小切口或天然孔洞（如腹腔镜手术、胸腔镜手术等）进行，使用特殊的手术器械和摄像头。手术紧急程度：可分为急诊手术、选择性手术和限期手术。手术目的：治疗性手术：用于治疗疾病，如切除肿瘤、修复损伤的组织等。诊断性手术：如活检，用于取得组织样本以确定疾病的性质。美容手术：用于改善外观，如隆胸、吸脂等。重建手术：如重建受损的肢体或面部。手术前，患者通常需要进行一系列的检查和评估，以确保他们适合进行手术，并减少手术风险。手术后，患者需要适当的恢复时间和护理，以促进伤口愈合和身体恢复。手术是现代医学的重要组成部分，能够治疗许多疾病和损伤，挽救生命，提高生活质量。随着技术的发展，手术技术也在不断进步，包括机器人辅助手术和远程手术等新兴领域

    """
    c="""
    解剖部位指的是在解剖学上对人体或动物体内各个结构的位置和关系的描述。解剖学是研究生物体结构的一个医学和生物学分支，它涉及对器官、组织、细胞甚至分子水平的结构的识别和描述。人体解剖学通常分为两个主要部分：宏观解剖学（或大体解剖学）：研究可以通过肉眼观察到的身体结构，如器官、骨骼、肌肉等。宏观解剖学又可以分为两个子领域：顶部解剖学：研究身体表面的结构。内部解剖学：研究身体内部的结构，包括器官的位置和相互关系。微观解剖学：研究无法用肉眼看到的小型结构，如细胞和组织。在医学和生物学中，了解和解剖部位的精确知识对于诊断疾病、规划手术、进行科学研究等都是至关重要的。解剖部位的命名通常遵循国际解剖学术语，以确保在全球范围内的一致性和准确性手术，也称为外科手术，是一种通过物理手段进入人体内部或表面，进行诊断、治疗或改善身体状况的医疗程序。手术通常由经过专业训练的外科医生（外科学家）或牙科外科医生执行，伴随着麻醉师提供的麻醉以减轻疼痛和保证患者的舒适度。
    """
    d="""
    药物是一种用于预防、诊断、治疗或缓解疾病症状的物质或组合物。药物可以通过不同的方式作用于人体，包括：改变或调节生理功能：例如，降压药物用于降低高血压，胰岛素用于调节血糖水平。抑制或杀灭病原体：例如，抗生素用于治疗细菌感染，抗病毒药物用于治疗病毒感染。缓解症状：例如，止痛药用于缓解疼痛，抗组胺药用于缓解过敏症状。替代或补充缺失的物质：例如，甲状腺激素替代疗法用于甲状腺功能减退症患者。影响心理状态和行为：例如，抗抑郁药用于治疗抑郁症，镇静剂用于治疗焦虑症。药物可以以多种形式存在，包括：固体：片剂、胶囊、粉末等。液体：溶液、悬浮液、注射液等。半固体：凝胶、乳膏、膏药等。药物的开发和使用需要严格的科学研究和监管审批过程。在药物上市前，通常需要经过临床前研究和临床试验，以证明其安全性、有效性和质量。药物的使用应遵循医生的处方和指导，以确保治疗效果和减少不良反应的风险。药物的分类可以根据其化学性质、作用机制、治疗用途等多种方式进行。正确和合理使用药物是现代医疗保健的重要组成部分，能够显著改善患者的生活质量和延长寿命解剖部位指的是在解剖学上对人体或动物体内各个结构的位置和关系的描述。解剖学是研究生物体结构的一个医学和生物学分支，它涉及对器官、组织、细胞甚至分子水平的结构的识别和描述。人体解剖学通常分为两个主要部分：宏观解剖学（或大体解剖学）：研究可以通过肉眼观察到的身体结构，如器官、骨骼、肌肉等。宏观解剖学又可以分为两个子领域：顶部解剖学：研究身体表面的结构。内部解剖学：研究身体内部的结构，包括器官的位置和相互关系。微观解剖学：研究无法用肉眼看到的小型结构，如细胞和组织。在医学和生物学中，了解和解剖部位的精确知识对于诊断疾病、规划手术、进行科学研究等都是至关重要的。解剖部位的命名通常遵循国际解剖学术语，以确保在全球范围内的一致性和准确性手术，也称为外科手术，是一种通过物理手段进入人体内部或表面，进行诊断、治疗或改善身体状况的医疗程序。手术通常由经过专业训练的外科医生（外科学家）或牙科外科医生执行，伴随着麻醉师提供的麻醉以减轻疼痛和保证患者的舒适度。
    """
    e="""
    影像检查是一系列用于观察和评估人体内部结构的非侵入性或微创性诊断技术。这些技术利用不同的物理原理来生成人体内部器官和组织的图像，从而帮助医生诊断疾病、监测疾病进展和评估治疗效果。常见的影像检查技术包括：X射线成像：使用X射线透过人体，根据不同组织对X射线的吸收差异来生成图像，常用于观察骨骼和某些软组织结构。计算机断层扫描（CT扫描）：通过旋转的X射线源和探测器获取一系列图像，然后由计算机处理这些图像以生成横截面图像，提供详细的器官和组织的三维信息。磁共振成像（MRI）：利用强磁场和无线电波产生人体内部的图像，特别适用于观察软组织和脑部，不使用有害的辐射。超声成像：使用高频声波来生成图像，常用于观察胎儿发育、心脏功能和腹部器官。核医学成像：包括单光子发射计算机断层扫描（SPECT）和正电子发射断层扫描（PET），使用放射性物质来评估器官和组织的功能和代谢活动。乳腺成像：包括乳房X射线摄影（钼靶）和超声成像，用于乳腺癌的筛查和诊断。影像检查在医疗诊断中起着至关重要的作用，因为它们能够提供关于内部结构的详细信息，帮助医生制定治疗计划。然而，影像检查并非没有风险，尤其是那些涉及辐射的技术，因此应该在医生的指导下合理使用药物是一种用于预防、诊断、治疗或缓解疾病症状的物质或组合物。]
    """ 
    f="""

    实验室检验，也称为临床实验室检验或医学实验室检验，是指通过对体液、组织或其他生物样本进行的一系列分析，以获取有关患者健康状况的信息。这些检验有助于疾病的诊断、治疗监控、疾病预防、健康状况评估和医学研究。实验室检验的范围非常广泛，包括但不限于以下几个方面：血液检验：检查血液样本中的红细胞、白细胞、血小板数量，以及血液的成分和生理功能，如血型、血糖、胆固醇、电解质等。尿液检验：分析尿液中的颜色、浓度、pH值、蛋白质、糖、红细胞、白细胞、细菌等，以评估肾脏功能和检测泌尿系统的疾病。生化检验：测定血液或尿液中的各种生化指标，如肝功能、肾功能、血糖、血脂、电解质、激素水平等。微生物检验：通过对体液、组织或其他样本的培养和鉴定，检测和识别病原微生物，如细菌、病毒、真菌和寄生虫。细胞学和病理学检：通过显微镜观察细胞形态和组织的结构，以诊断癌症、炎症、感染等疾病。分子检验：利用分子生物学技术，如PCR（聚合酶链反应），检测DNA、RNA或蛋白质，用于诊断遗传性疾病、感染性疾病和肿瘤等。免疫学检验：检测体内的免疫反应和特定抗体或抗原的存在，用于诊断自身免疫性疾病、过敏反应和疫苗接种后的免疫状态。实验室检验通常由专业的医疗技术人员在临床实验室进行，他们使用各种仪器和试剂来分析样本。检验结果由医生或临床病理学家解释，并与患者的临床症状和其他检查结果一起用于医疗决策。
    """
    #a_sents_train, b_sents_train, scores_train = load_dataset(TRAIN_PATH, 'train')
    #a_sents_test, b_sents_test, scores_test = load_dataset(TEST_PATH, 'test')
    #a_sents_dev, b_sents_dev, scores_dev = load_dataset(DEV_PATH, 'dev')
    #print("Loading {} training samples from {}".format(len(scores_train), TRAIN_PATH))
    #print("Loading {} developing samples from {}".format(len(scores_dev), DEV_PATH))
    #print("Loading {} testing samples from {}".format(len(scores_test), TEST_PATH))

    tokenizer, model = build_model(MODEL_NAME)
    print("Building {} tokenizer and model successfuly.".format(MODEL_NAME))

    print("Transfer sentences to BERT vectors.")
    a_vecs = sents_to_vecs(a, tokenizer, model)
    b_vecs = sents_to_vecs(b, tokenizer, model)
    c_vecs = sents_to_vecs(c, tokenizer, model)
    d_vecs = sents_to_vecs(d, tokenizer, model)
    e_vecs = sents_to_vecs(e, tokenizer, model)
    f_vecs = sents_to_vecs(f, tokenizer, model)
    c1_vecs = sents_to_vecs(c1, tokenizer, model)
    c2_vecs = sents_to_vecs(c2, tokenizer, model)
    c3_vecs = sents_to_vecs(c3, tokenizer, model)
    c4_vecs = sents_to_vecs(c4, tokenizer, model)
    c5_vecs = sents_to_vecs(c5, tokenizer, model)
    c6_vecs = sents_to_vecs(c6, tokenizer, model)
    c7_vecs = sents_to_vecs(c7, tokenizer, model)
    c8_vecs = sents_to_vecs(c8, tokenizer, model)
    c9_vecs = sents_to_vecs(c9, tokenizer, model)

    if USE_WHITENING:

        print("Compute kernel and bias.")
        a_train=_read_json_text('datasets/cnmer/train.txt')
        a_dev=_read_json_text('datasets/cnmer/dev.txt')
        at_vecs = sents_to_vecs(a_train, tokenizer, model)
        ad_vecs = sents_to_vecs(a_dev, tokenizer, model)
        kernel, bias = compute_kernel_bias([
            a_vecs, b_vecs,c_vecs,d_vecs,e_vecs,f_vecs,at_vecs,ad_vecs,c1_vecs,c2_vecs,c3_vecs,c4_vecs,c5_vecs,c6_vecs,c7_vecs,c8_vecs,c9_vecs
        ], n_components=N_COMPONENTS)
        '''
        a_vecs = transform_and_normalize(a_vecs, kernel, bias)
        b_vecs = transform_and_normalize(b_vecs, kernel, bias)
        c_vecs = transform_and_normalize(c_vecs, kernel, bias)
        d_vecs = transform_and_normalize(d_vecs, kernel, bias)
        e_vecs = transform_and_normalize(e_vecs, kernel, bias)
        f_vecs = transform_and_normalize(f_vecs, kernel, bias)
        '''
        c1_vecs = transform_and_normalize(c1_vecs, kernel, bias)
        c2_vecs = transform_and_normalize(c2_vecs, kernel, bias)
        c3_vecs = transform_and_normalize(c3_vecs, kernel, bias)
        c4_vecs = transform_and_normalize(c4_vecs, kernel, bias)
        c5_vecs = transform_and_normalize(c5_vecs, kernel, bias)
        c6_vecs = transform_and_normalize(c6_vecs, kernel, bias)
        c7_vecs = transform_and_normalize(c7_vecs, kernel, bias)
        c8_vecs = transform_and_normalize(c8_vecs, kernel, bias)
        c9_vecs = transform_and_normalize(c9_vecs, kernel, bias)
    else:
        a_vecs = normalize(a_vecs)
        b_vecs = normalize(b_vecs)
    '''
    av=np.mean(a_vecs,0).astype(float)
    bv=np.mean(b_vecs,0).astype(float)
    cv=np.mean(c_vecs,0).astype(float)
    dv=np.mean(d_vecs,0).astype(float)
    ev=np.mean(e_vecs,0).astype(float)
    fv=np.mean(f_vecs,0).astype(float)
    
    print(a_vecs.shape,b_vecs.shape,av.shape,bv.shape,cv.shape,dv.shape,ev.shape,fv.shape)
    np.save('./maa.npy',av)
    np.save('./mab.npy',bv)
    np.save('./mac.npy',cv)
    np.save('./mad.npy',dv)
    np.save('./mae.npy',ev)
    np.save('./maf.npy',fv)
    '''
    cv1=np.mean(c1_vecs,0).astype(float)
    cv2=np.mean(c2_vecs,0).astype(float)
    cv3=np.mean(c3_vecs,0).astype(float)
    cv4=np.mean(c4_vecs,0).astype(float)
    cv5=np.mean(c5_vecs,0).astype(float)
    cv6=np.mean(c6_vecs,0).astype(float)
    cv7=np.mean(c7_vecs,0).astype(float)
    cv8=np.mean(c8_vecs,0).astype(float)
    cv9=np.mean(c9_vecs,0).astype(float)
    np.save('./c1.npy',cv1)
    np.save('./c2.npy',cv2)
    np.save('./c3.npy',cv3)
    np.save('./c4.npy',cv4)
    np.save('./c5.npy',cv5)
    np.save('./c6.npy',cv6)
    np.save('./c7.npy',cv7)
    np.save('./c8.npy',cv8)
    np.save('./c9.npy',cv9)
    print("Results:")


if __name__ == "__main__":
    main()

