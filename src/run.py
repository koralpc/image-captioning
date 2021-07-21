from src.model import EDModel
from src.train import Trainer
from dataset import ImageCaptionDataset


annotation_url = (
    "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
)
img_url = "http://images.cocodataset.org/zips/train2014.zip"
buffer_size = 1000
limit_size = 100
batch_size = 64
embedding_dim = 256
units = 512
top_k = 5000
vocab_size = top_k + 1
num_epochs = 20
# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
features_shape = 2048
attention_features_shape = 64
checkpoint_dir = "./checkpoints/train"
caption_dataset = ImageCaptionDataset(img_url, annotation_url)
train_data, val_data, max_len, tokenizer = caption_dataset.prepare_data(limit_size, buffer_size, batch_size)
model = EDModel(embedding_dim,units,vocab_size,tokenizer)
train_config = dict(buffer_size=buffer_size,limit_size=limit_size,batch_size=batch_size,max_length=max_len,attn_shape=attention_features_shape)
trainer = Trainer(checkpoint_path=checkpoint_dir)

def main():
    num_steps = len(train_data) // batch_size
    ckpt,ckpt_manager = trainer.set_checkpoint(model)
    trainer.train(model,train_data,ckpt,ckpt_manager,num_epochs,num_steps)