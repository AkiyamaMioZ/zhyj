import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from scipy.stats import mode
import jieba
import joblib
import pickle
import warnings
import random
warnings.filterwarnings("ignore")



negative_responses = [
    "垃圾", "差劲", "太差了", "不好玩", "无聊", "难玩", "bug多", "卡顿", "坑", "氪金",
    "太贵", "不值", "坑钱", "骗钱", "烂", "糟糕", "失望", "后悔", "浪费时间", "无趣",
    "枯燥", "乏味", "无聊", "恶心", "反感", "讨厌", "令人生气", "气愤", "生气", "愤怒",
    
    "操作难", "画质差", "优化差", "掉帧严重", "老是卡死", "经常闪退", "总是崩溃", "网络卡顿", "延迟高", "匹配太慢",
    "外挂泛滥", "作弊严重", "开挂现象多", "服务器不稳定", "排队时间长", "没有新意", "设计无脑", "套路太多", "抽卡坑人", "太肝了",
    
    "关卡设计差", "手感很差", "打击感不足", "手感生涩", "操作不顺", "反应迟钝", "打击感太弱", "手感不协调",
    "剧情尴尬", "剧情脱节", "剧情混乱", "剧情崩坏", "剧情别扭", "剧情单薄", "剧情平淡", "剧情俗套",
    
    "更新拖沓", "更新质量差", "维护不及时", "客服态度差", "社区氛围差", "玩家流失严重", "氪金压力大", "平衡性差", "福利太少",
    "活动单调", "内容匮乏", "玩法单一", "重复刷刷刷", "任务无聊", "奖励太少", "设计死板",
    
    "机制陈旧", "机制落后", "机制混乱", "机制不合理", "机制死板", "机制单调", "机制重复", "机制割裂",
    "系统臃肿", "系统混乱", "系统不合理", "系统僵化", "系统死板", "系统单调", "系统割裂"
]

positive_responses = [
    "好玩", "有趣", "精彩", "优秀", "太棒了", "非常不错", "让人着迷", "值得推荐", "特别值得", "超级好评",
    "画质精美", "音效震撼", "剧情出色", "玩法新颖", "很有创意", "独具特色", "精美绝伦", "令人震撼", "特别感人", "令人惊艳",
    "上头", "沉浸感强", "享受", "快乐", "开心", "欢乐", "刺激", "过瘾", "爽快", "带感十足",
    
    "优化完美", "特别流畅", "非常稳定", "丝般顺滑", "体验舒适", "很良心", "很用心", "很有诚意", "很走心", "制作精良",
    "运行流畅", "画面稳定", "加载神速", "响应迅速", "网络顺畅", "延迟极低", "匹配迅速", "体验顺畅",
    
    "关卡设计巧妙", "手感绝佳", "打击感十足", "操作顺畅", "反应灵敏", "手感细腻", "打击感强烈", "操控精准",
    "剧情精彩", "剧情感人", "剧情出色", "剧情动人", "剧情吸引人", "剧情引人入胜", "剧情扣人心弦", "剧情新颖独特",
    
    "更新及时", "更新给力", "维护到位", "客服贴心", "社区活跃", "玩家众多", "氪金合理", "平衡性好", "福利给力",
    "活动丰富", "内容充实", "玩法多样", "任务有趣", "奖励丰厚", "设计巧妙", "体验顺畅"
]
response_templates = {
        0: [  # 负面回复模板
            "这游戏确实有点{}，不过开发团队应该会听取玩家意见的",
            "玩起来感觉{}，希望之后的版本能改进一下",
            "确实存在{}的问题，但相信会慢慢优化的",
            "游戏目前还比较{}，期待后续更新能够改善",
            "作为老玩家也觉得有点{}，建议开发者多听听玩家的声音",
            "玩了一段时间发现挺{}的，希望能尽快修复这些问题",
            "游戏内容有些{}，但瑕不掩瑜，还是值得期待的",
            "最近更新后变得有点{}，希望能尽快调整回来",
            "确实有一些{}的地方，但瑕不掩瑜吧",
            "虽然有点{}，但相信开发团队会努力改进的"
        ],
        1: [  # 正面回复模板
            "玩了好久，真的太{}了，强烈推荐大家试试",
            "这游戏是真的{}，玩得停不下来啊",
            "不得不说，游戏体验非常{}，让人欲罢不能",
            "最近玩的最{}的游戏，没有之一！",
            "游戏品质相当{}，很久没玩到这么好的游戏了",
            "每次玩都感觉特别{}，真是难得的好游戏",
            "这游戏太{}了，我已经推荐给身边的朋友了",
            "玩起来真的很{}，完全符合我的期待",
            "不愧是大家都说{}的游戏，确实名不虚传",
            "作为一个老玩家，这游戏是真的{}"
        ]
    }
# 下载必要的中文停用词
stop_words = set(stopwords.words('chinese')) if 'chinese' in stopwords.fileids() else set()


# 加载中文标点符号
punctuations = set(string.punctuation).union({'，', '。', '、', '！', '？', '；', '：', '（', '）', '【', '】', '《', '》', '“', '”', '‘', '’'})

# 文本清洗和分词
def preprocess_text(text):
    # 使用 jieba 分词
    words = jieba.lcut(text)
    # 去除停用词和标点符号
    cleaned_words = [word for word in words if word not in stop_words and word not in punctuations]
    return " ".join(cleaned_words)

# 训练
def train():
    # 加载数据
    data = pd.read_csv("taptap_review_ready.csv")

    # 数据清洗
    data['cleaned_text'] = data['review'].apply(preprocess_text)

    # TF-IDF 向量化
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(data['cleaned_text']).toarray()
    y = data['sentiment']

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. 朴素贝叶斯模型
    print("朴素贝叶斯模型训练中...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    nb_pred = nb_model.predict(X_test)
    print("朴素贝叶斯模型训练完成")

    # 2. SVM 模型（使用网格搜索找到最佳参数）
    print("SVM 模型训练中...")
    svm_model = SVC(probability=True, random_state=42)
    param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    svm_grid = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy')
    svm_grid.fit(X_train, y_train)
    svm_model = svm_grid.best_estimator_
    svm_pred = svm_model.predict(X_test)
    print("SVM 模型训练完成")

    # 3. XGBoost 模型
    print("XGBoost 模型训练中...")
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    print("XGBoost 模型训练完成")
    # 保存模型和向量化器
    joblib.dump(nb_model, 'nb_model.pkl')
    joblib.dump(svm_model, 'svm_model.pkl')
    joblib.dump(xgb_model, 'xgb_model.pkl')
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    print("模型和向量化器已保存。")
    # 4. 模型融合（简单投票法）
    # 将各模型的预测结果进行投票
    predictions = pd.DataFrame({
        'NB': nb_pred,
        'SVM': svm_pred,
        'XGB': xgb_pred
    })

    # 通过取众数决定最终预测结果
    final_pred = mode(predictions, axis=1).mode.flatten()

    # 5. 评估性能
    print("朴素贝叶斯模型：")
    print("Accuracy:", accuracy_score(y_test, nb_pred))
    print("Classification Report:\n", classification_report(y_test, nb_pred))

    print("SVM 模型：")
    print("Accuracy:", accuracy_score(y_test, svm_pred))
    print("Classification Report:\n", classification_report(y_test, svm_pred))

    print("XGBoost 模型：")
    print("Accuracy:", accuracy_score(y_test, xgb_pred))
    print("Classification Report:\n", classification_report(y_test, xgb_pred))

    print("融合模型：")
    print("Accuracy:", accuracy_score(y_test, final_pred))
    print("Classification Report:\n", classification_report(y_test, final_pred))


def load_model():
    from pathlib import Path
    import os
    # 获取当前文件的路径
    current_file_path = Path(__file__).resolve()
    nb_model = joblib.load(os.path.join(current_file_path.parent,'model/nb_model.pkl'))
    svm_model = joblib.load(os.path.join(current_file_path.parent,'model/svm_model.pkl'))
    xgb_model = joblib.load(os.path.join(current_file_path.parent,'model/xgb_model.pkl'))
    with open(os.path.join(current_file_path.parent,'model/tfidf_vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)
    return nb_model, svm_model, xgb_model, vectorizer
# 加载模型和向量化器
    
nb_model, svm_model, xgb_model, vectorizer = load_model()
def predict(inputs):
    # 使用加载的模型进行预测
    data = [preprocess_text(inputs)]
    new_comments_vectorized = vectorizer.transform(data).toarray()

    nb_new_pred = nb_model.predict(new_comments_vectorized)
    svm_new_pred = svm_model.predict(new_comments_vectorized)
    xgb_new_pred = xgb_model.predict(new_comments_vectorized)

    # 融合预测
    new_predictions = pd.DataFrame({
        'NB': nb_new_pred,
        'SVM': svm_new_pred,
        'XGB': xgb_new_pred
    })
    final_new_pred = mode(new_predictions, axis=1).mode.flatten()
    response = generate_response(final_new_pred[0])
    return response

def generate_response(sentiment):
    """根据情感预测生成响应"""

    
    if sentiment == 0:
        word = random.choice(negative_responses)
        template = random.choice(response_templates[0])
    elif sentiment == 1:
        word = random.choice(positive_responses)
        template = random.choice(response_templates[1])
    else:
        return "只能是[0,1]"
    
    return template.format(word)






if __name__ == '__main__':

    response = predict("太好玩好玩")
    print(response)