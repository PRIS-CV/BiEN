import sys
import os
import copy
import torch
import yaml
sys.path.append('../../../../')
from models.BiFRN_snapshot import BiFRN
from utils import util
from trainers.eval_snapshot_n import meta_test


with open('../../../../config.yml', 'r') as f:
    temp = yaml.safe_load(f)
data_path = os.path.abspath(temp['data_path'])

test_path = os.path.join(data_path,'CUB_fewshot_cropped/test_pre')

gpu = 3
torch.cuda.set_device(gpu)


stage = 3
epoch = 4


model = BiFRN(resnet=True)
model.cuda()
models = [copy.deepcopy(model) for i in range(stage)]

[models[i].load_state_dict(torch.load(str(epoch*(i+1))+'_'+'model_ResNet-12.pth')) for i in range(stage)]



with torch.no_grad():
    way = 5
    for shot in [1,5]:
        mean,interval = meta_test(data_path=test_path,
                                models=models,
                                way=way,
                                shot=shot,
                                pre=True,
                                transform_type=None,
                                trial=10000)
        print('%d-way-%d-shot acc: %.3f\t%.3f'%(way,shot,mean,interval))
