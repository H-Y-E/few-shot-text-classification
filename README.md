# few-shot-text-classification_CCF-baseline
### BERT模型
下载pytorch_model.bin放到input/bert-base-chinese目录下<br>
链接：https://pan.baidu.com/s/1EfGn9pAd_mbmAB_bZrWHZg 
提取码：mpis

### 基于TF-IDF的词替换增强后的数据集
每条数据基于所在数据集abstract部分构成的语料库，替换其abstract中的五个词，生成一条句子。生成文件中源语句与其增强后的语句紧邻<br>
* new_train_TF.json(1950条)<br>
* mytrain_TF.json(1743条)<br>

> eg:<br>
> s1<br>
> s1-aug<br>
> s2<br>
> s2-aug

### 训练效果
|  训练集  | f1值  |
|  ----  | ----  |
| new_train.json | 0.6784743670571368 |
| new_train_TF.json |0.686600732901836 (↑ 0.0082)|
| mytrain.json  |0.5921483873650565|
|mytrain_TF.json|0.6283421097079 (↑ 0.03619)|
