from training.data import get_wds_dataset
# from open_clip import get_tokenizer
# from ..training import get_tokenizer
from open_clip.factory import get_tokenizer
from open_clip.transform import image_transform
from megatron.data.vit_dataset import ClassificationTransform
import argparse
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

args = parser.parse_args()
args.train_data = "/mnt/zoltan/zanzong/CC3M/cc3m/{00000..00331}.tar"
args.train_num_samples = 1000000
args.train_data_upsampling_factors = None
args.seed = 1234
args.batch_size = 16
args.workers = 1
args.world_size = 1
args.model = 'RN50'

def preprocess_img(img):
    return img 

def text_tokenizor(text):
    #get_tokenizer(args.model)
    return [text]

img_transform = ClassificationTransform((256, 256))

data = {}
data['train'] = get_wds_dataset(args, img_transform, True, tokenizer=get_tokenizer(args.model))
print(data)
data['train'].set_epoch(0)
dataloader = data['train'].dataloader
data_iterater = iter(dataloader)
data = next(data_iterater)
text_data = data[1]
eod_token = 49408
print(len(data[0]), data[1].shape)
# print(text_data[:5][:20])
print(text_data == eod_token)
# for i, batch in enumerate(dataloader):
#     images, texts = batch
#     print(images[0], texts[0])
#     if i > 3:
#         break
