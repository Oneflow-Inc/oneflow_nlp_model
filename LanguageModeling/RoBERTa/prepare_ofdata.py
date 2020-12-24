import oneflow.core.record.record_pb2 as ofrecord
import six
import random
import struct
from transformers import GlueDataset, GlueDataTrainingArguments, RobertaTokenizer
import sys, os

data_dir = sys.argv[1] # glue_dir
task_name = sys.argv[2] # MRPC
output_dir=sys.argv[3] # out_dir

glue_args = GlueDataTrainingArguments(
    task_name=task_name,
    data_dir=os.path.join(data_dir, task_name),
    max_seq_length=128,
    overwrite_cache=False
)
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def int32_feature(value):
    if not isinstance(value, (list, tuple)):
        value = [value]
    return ofrecord.Feature(int32_list=ofrecord.Int32List(value=value))


def int64_feature(value):
    if not isinstance(value, (list, tuple)):
        value = [value]
    return ofrecord.Feature(int64_list=ofrecord.Int64List(value=value))


def float_feature(value):
    if not isinstance(value, (list, tuple)):
        value = [value]
    return ofrecord.Feature(float_list=ofrecord.FloatList(value=value))


def double_feature(value):
    if not isinstance(value, (list, tuple)):
        value = [value]
    return ofrecord.Feature(double_list=ofrecord.DoubleList(value=value))


def bytes_feature(value):
    if not isinstance(value, (list, tuple)):
        value = [value]
    if not six.PY2:
        if isinstance(value[0], str):
            value = [x.encode() for x in value]
    return ofrecord.Feature(bytes_list=ofrecord.BytesList(value=value))

def convert_to_pack(item):
    pack = {
        'input_ids': item.input_ids,
        'input_mask': item.attention_mask if item.attention_mask else [1 for _ in range(glue_args.max_seq_length)],
        'segment_ids': item.token_type_ids if item.token_type_ids else [0 for _ in range(glue_args.max_seq_length)],
        'label_ids': item.label,
        'is_real_example': 1,
    }
    for name, val in pack.items():
        pack[name] = int64_feature(val)
    return pack

def convert_to_ofrecord(dataset, split):
    fn = os.path.join(output_dir, task_name, split, '{}.of_record-0'.format(split))
    os.makedirs(os.path.dirname(fn), exist_ok=True)

    with open(fn, 'wb') as f:
        for i in range(len(dataset)):
            topack = convert_to_pack(dataset[i])
            ofrecord_features = ofrecord.OFRecord(feature=topack)
            serilizedBytes = ofrecord_features.SerializeToString()

            length = ofrecord_features.ByteSize()

            f.write(struct.pack("q", length))
            f.write(serilizedBytes)

def get_of_dataset():
    train_data = GlueDataset(glue_args, tokenizer, mode='train')
    eval_data = GlueDataset(glue_args, tokenizer, mode='dev')
    print('train: {}, eval: {}'.format(len(train_data), len(eval_data)))
    convert_to_ofrecord(train_data, 'train')
    convert_to_ofrecord(eval_data, 'eval')

if __name__ == '__main__':
    get_of_dataset()
