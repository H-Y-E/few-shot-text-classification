## few-shot-text-classification
### Model结构
kun哥修改后的CCF_baseline(增添了lstm层)
### BERT模型
下载pytorch_model.bin放到input/bert-base-chinese目录下<br>
链接：https://pan.baidu.com/s/1EfGn9pAd_mbmAB_bZrWHZg <br>
提取码：mpis

### 基于TF-IDF的词替换增强后的数据集
每条数据基于所在数据集abstract部分构成的语料库，替换其abstract中的五个词，生成一条句子。生成文件中原语句与其对应增强后的语句紧邻。<br>
* [new_train_TF.json(1950条)](https://github.com/H-Y-E/few-shot-text-classification_CCF-baseline/tree/main/few-shot-text-classificatoin/input/data_aug)<br>
* [mytrain_TF.json(1743条)](https://github.com/H-Y-E/few-shot-text-classification_CCF-baseline/tree/main/few-shot-text-classificatoin/input/data_aug)<br>

> eg:<br>
> s1<br>
> s1-aug<br>
> s2<br>
> s2-aug

### 训练效果
|  训练集  | F1_macro  |
|  ----  | ----  |
| new_train.json | 0.67847 |
| new_train_TF.json |0.68660 (↑ 0.0082)|
| mytrain.json  |0.59214|
|mytrain_TF.json|0.62834 (↑ 0.0362)|

### 基于新baseline的数据增强效果
[Baseline_From] : https://github.com/Hyman25/few-shot-text-classification-baseline<br>
[Data_And_Performance] ：
| 数据 | 说明 | 验证集_f1_macro | [测试集](https://github.com/H-Y-E/few-shot-text-classification/blob/main/few-shot-text-classificatoin/input/new_test.json) |
| :----: | :----: | :----: | :----: |
| new_train.json | 元数据 |   | 0.68 |
| [new_train_TF.json](https://github.com/H-Y-E/few-shot-text-classification/blob/main/few-shot-text-classificatoin/input/data_aug/new_train_TF.json) | 1:1生成增强数据 | 0.9579 | 0.6892(↑0.0092) |
| [new_train_TF_3.json](https://github.com/H-Y-E/few-shot-text-classification/blob/main/few-shot-text-classificatoin/input/data_aug/new_train_TF_3.json) | 1:3生成增强数据 | 1.0 | 0.6902(↑0.0102) |
