#忽略由于pandas摒弃append造成的频繁报错
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from transformers import BertTokenizer, AutoModel, AdamW, AutoConfig
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
import torch.nn.functional as F
from tqdm import tqdm
import copy
import torch.nn as nn
import os
from torch.autograd import Variable
# 如果有多卡可以指定使用哪张卡进行训练
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# 模型训练参数设置
CONFIG = {
    'fold': 10,
    'model_path': "../input/bert-base-chinese",
    'data_path': "../input/data_aug/new_train_TF.json",
    'max_length': 512,
    'train_batchsize': 4,
    'vali_batchsize': 4,
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "learning_rate": 1e-5,
    "min_lr": 1e-6,
    "weight_decay": 1e-6,
    "T_max": 500,
    "seed": 42,
    "num_class": 32,
    #"num_class": 16,
    "epoch_times": 50,
}


# 设置随机种子
def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)



set_seed(CONFIG['seed'])


class ModelConfig:
    batch_size = 32
    output_size = 2
    hidden_dim = 384  # 768/2
    n_layers = 2
    lr = 2e-5
    bidirectional = True  # 这里为True，为双向LSTM
    # training params
    epochs = 10
    # batch_size=50
    print_every = 10
    clip = 5  # gradient clipping
    bert_path = 'bert-base-chinese'  # 预训练bert路径
    save_path = 'bert_bilstm.pth'  # 模型保存路径


# 读取训练文件
with open(CONFIG['data_path'], "r", encoding='UTF-8') as f:
    file_data = f.readlines()

#df = pd.DataFrame(columns=['title', 'assignee', 'abstract', 'label_id'])
df = pd.DataFrame(columns=['id', 'title', 'assignee', 'abstract', 'label_id'])
for each_json in file_data:
    json_dict = eval(each_json)
    df = df.append(json_dict, ignore_index=True)

df["label_id"] = df["label_id"].astype(int)

# 根据KFOLD划分数据
gkf = StratifiedKFold(n_splits=CONFIG['fold'])

for fold, (_, val_) in enumerate(gkf.split(X=df, y=df.label_id)):
    df.loc[val_, "kfold"] = int(fold)

df["kfold"] = df["kfold"].astype(int)
df.groupby('kfold')['label_id'].value_counts()


class CCFDataSet(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.id = df["id"].values
        self.title = df["title"].values
        self.assignee = df["assignee"].values
        self.abstract = df["abstract"].values
        self.label_id = df["label_id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        data_id = self.id[index]
        title = self.title[index]
        assignee = self.assignee[index]
        abstract = self.abstract[index]
        label = self.label_id[index]

        text = "这份专利的标题为：《{}》，由“{}”公司申请，详细说明如下：{}".format(title, assignee, abstract)
        inputs = self.tokenizer.encode_plus(text, truncation=True, add_special_tokens=True, max_length=self.max_length)

        return {'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask'],
                'label': label}


class CCFModel(nn.Module):

    def __init__(self,hidden_dim, output_size, n_layers, bidirectional=True, drop_prob=0.5):
        super(CCFModel, self).__init__()
        self.model = AutoModel.from_pretrained(CONFIG['model_path'])
        self.config = AutoConfig.from_pretrained(CONFIG['model_path'])
        self.dropout = nn.Dropout(drop_prob)
        self.lstm = nn.LSTM(768, hidden_dim, n_layers, batch_first=True, bidirectional=bidirectional)
        self.norm1=nn.LayerNorm(hidden_dim * 2)
        self.norm2=nn.LayerNorm(256)
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.fc2 = nn.Linear(256, 36)
        if bidirectional:
            self.fc1 = nn.Linear(hidden_dim * 2, 256)
        else:
            self.fc1 = nn.Linear(hidden_dim, 36)
        self._init_weights(self.fc1)
        self._init_weights(self.fc2)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        number = 1
        if self.bidirectional:
            number = 2

        #修改                      !!
        #hidden = (weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float().cuda(),
        #        weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float().cuda()
        #              )
        hidden = (weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float(),
                weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float()
                      )

        return hidden

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, ids, h,mask):
        batch_size = ids.size(0)
        x = self.model(ids,mask)[0]
        lstm_out, (hidden_last, cn_last) = self.lstm(x, h)
        if self.bidirectional:
            # 正向最后一层，最后一个时刻
            hidden_last_L = hidden_last[-2]
            # print(hidden_last_L.shape)  #[32, 384]
            # 反向最后一层，最后一个时刻
            hidden_last_R = hidden_last[-1]
            # print(hidden_last_R.shape)   #[32, 384]
            # 进行拼接
            hidden_last_out = torch.cat([hidden_last_L, hidden_last_R], dim=-1)
            # print(hidden_last_out.shape,'hidden_last_out')   #[32, 768]
        else:
            hidden_last_out = hidden_last[-1]  # [32, 384]
        out = self.norm1(hidden_last_out)
        out = self.fc1(out)
        out = self.norm2(out)
        # print(out.shape)    #[32,768]
        out = self.fc2(out)

        return out



class Collate():
    def __init__(self, tokenizer, isTrain=True):
        self.tokenizer = tokenizer
        self.isTrain = isTrain

    def __call__(self, batch):
        output = dict()
        output['input_ids'] = [sample['input_ids'] for sample in batch]
        output['attention_mask'] = [sample['attention_mask'] for sample in batch]

        if self.isTrain:
            output['label'] = [sample['label'] for sample in batch]

        btmax_len = max([len(i) for i in output['input_ids']])

        # 手动进行pad填充
        if self.tokenizer.padding_side == 'right':
            output['input_ids'] = [i + [self.tokenizer.pad_token_id] * (btmax_len - len(i)) for i in
                                   output['input_ids']]
            output['attention_mask'] = [i + [0] * (btmax_len - len(i)) for i in output['attention_mask']]
        else:
            output['input_ids'] = [[self.tokenizer.pad_token_id] * (btmax_len - len(i)) + i for i in
                                   output['input_ids']]
            output['attention_mask'] = [[0] * (btmax_len - len(i)) + i for i in output['attention_mask']]

        output['input_ids'] = torch.tensor(output['input_ids'], dtype=torch.long)
        output['attention_mask'] = torch.tensor(output['attention_mask'], dtype=torch.long)

        if self.isTrain:
            output['label'] = torch.tensor(output['label'], dtype=torch.long)

        return output


def criterion(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)


def get_labels(outputs, labels):
    # pred_labels 和 true_labels 便于后续计算F1分数
    outputs = F.softmax(outputs, dim=1).cpu().numpy()
    pred_labels = outputs.argmax(1)
    pred_labels = pred_labels.tolist()

    true_labels = labels.cpu().tolist()
    return pred_labels, true_labels


def train_one_epoch(model, train_loader, optimizer, scheduler, epoch, device):
    model.train()
    dataset_size = 0
    running_loss = 0.0

    bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, data in bar:
        ids = data['input_ids'].to(device, dtype=torch.long)
        mask = data['attention_mask'].to(device, dtype=torch.long)
        labels = data['label'].to(device, dtype=torch.long)
        h = model.init_hidden(4)
        h = tuple([each.data for each in h])
        batch_size = ids.size(0)
        outputs = model(ids, h,mask)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch, Train_loss=epoch_loss, LR=optimizer.param_groups[0]['lr'])
    return epoch_loss


@torch.no_grad()
def valid_one_epoch(model, vali_loader, optimizer, epoch, device):
    model.eval()
    dataset_size = 0
    running_loss = 0.0

    pred_labels = []
    true_labels = []

    bar = tqdm(enumerate(vali_loader), total=len(vali_loader))
    for step, data in bar:
        ids = data['input_ids'].to(device, dtype=torch.long)
        mask = data['attention_mask'].to(device, dtype=torch.long)
        labels = data['label'].to(device, dtype=torch.long)
        h = model.init_hidden(4)
        h = tuple([each.data for each in h])
        
        batch_size = ids.size(0)
        outputs = model(ids, h,mask)

        loss = criterion(outputs, labels)

        batch_pred_labels, true_pred_labels = get_labels(outputs, labels)
        pred_labels += batch_pred_labels
        true_labels += true_pred_labels
        epoch_score = f1_score(pred_labels, true_labels, average='macro')

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch, Valid_loss=epoch_loss, F_score=epoch_score, LR=optimizer.param_groups[0]['lr'])
    return epoch_loss, epoch_score


def main():
    tokenizer = BertTokenizer.from_pretrained(CONFIG['model_path'])
    print('------------------')
    print('load model')
    print(tokenizer)
    print('-----------------------')
    model_config = ModelConfig()
    collate_fn = Collate(tokenizer, True)

    # 拆分训练集和验证集,默认第一个fold作为验证集，后九个为训练集，则训练集占90%，验证集占10%
    fold = 0
    train_data = df[df["kfold"] != fold].reset_index(drop=True)
    vali_data = df[df["kfold"] == fold].reset_index(drop=True)

    train_dataset = CCFDataSet(train_data, tokenizer, CONFIG['max_length'])
    vali_dataset = CCFDataSet(vali_data, tokenizer, CONFIG['max_length'])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['train_batchsize'], collate_fn=collate_fn,
                              shuffle=True, drop_last=True, pin_memory=False, num_workers=8)
    vali_loader = DataLoader(vali_dataset, batch_size=CONFIG['vali_batchsize'], collate_fn=collate_fn,
                             shuffle=False, pin_memory=False, num_workers=8)

    model = CCFModel(hidden_dim=model_config.hidden_dim,output_size=model_config.output_size,n_layers=model_config.n_layers,bidirectional=True,drop_prob=0.1)
    model.to(CONFIG['device'])

    optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['T_max'], eta_min=CONFIG['min_lr'])

    # 数据和参数准备结束，开始训练
    if torch.cuda.is_available():
        print("GPU: {}\n".format(torch.cuda.get_device_name()))

    best_weights = copy.deepcopy(model.state_dict())
    best_score = 0

    for epoch in range(CONFIG['epoch_times']):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, epoch, CONFIG['device'])
        valid_loss, valid_score = valid_one_epoch(model, vali_loader, optimizer, epoch, CONFIG['device'])

        if valid_score >= best_score:
            print(f"Validation Score Improved ({best_score} ---> {valid_score})")
            best_score = valid_score
            best_weights = copy.deepcopy(model.state_dict())

            PATH = f"best_weights.bin"
            torch.save(model.state_dict(), PATH)

    print("Best F1 score:" + str(best_score))

def test():
    tokenizer = BertTokenizer.from_pretrained(CONFIG['model_path'])
    print('------------------')
    print('load model')
    print(tokenizer)
    print('-----------------------')
    model_config = ModelConfig()
    collate_fn = Collate(tokenizer, True)

    # 拆分训练集和验证集,默认第一个fold作为验证集，后九个为训练集，则训练集占90%，验证集占10%
    fold = 0
    #train_data = df[df["kfold"] != fold].reset_index(drop=True)
    vali_data = df[df["kfold"] == fold].reset_index(drop=True)

    #train_dataset = CCFDataSet(train_data, tokenizer, CONFIG['max_length'])
    vali_dataset = CCFDataSet(vali_data, tokenizer, CONFIG['max_length'])

    #train_loader = DataLoader(train_dataset, batch_size=CONFIG['train_batchsize'], collate_fn=collate_fn,
    #                          shuffle=True, drop_last=True, pin_memory=False, num_workers=8)
    vali_loader = DataLoader(vali_dataset, batch_size=CONFIG['vali_batchsize'], collate_fn=collate_fn,
                             shuffle=False, pin_memory=False, num_workers=8)

    model = CCFModel(hidden_dim=model_config.hidden_dim,output_size=model_config.output_size,n_layers=model_config.n_layers,bidirectional=True,drop_prob=0.1)
    ckpt = torch.load('./best_weights.bin', map_location=CONFIG['device'])
    ckpt.pop('fc2.bias')
    ckpt.pop('fc2.weight')
    model.load_state_dict(ckpt, strict=False)
    #model.to(CONFIG['device'])

    optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['T_max'], eta_min=CONFIG['min_lr'])

    # 数据和参数准备结束，开始训练
    if torch.cuda.is_available():
        print("GPU: {}\n".format(torch.cuda.get_device_name()))

    #best_weights = copy.deepcopy(model.state_dict())
    #best_score = 0

    valid_loss, valid_score = valid_one_epoch(model, vali_loader, optimizer, 1, CONFIG['device'])

    #输出结果
    print("valid_loss : {} valid_score: {}". format(valid_loss, valid_score))


if __name__ == '__main__':
    main()
    #test()