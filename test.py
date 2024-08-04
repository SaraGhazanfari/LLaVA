import os

from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model

model_base = None
model_path = os.path.expanduser('/scratch/sg7457/code/LLaVA/checkpoints/llava-v1.7-7b')
model_name = get_model_name_from_path(model_path)
tokenizer, v2_model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)


model_base = None
model_path = os.path.expanduser('/scratch/sg7457/code/LLaVA/checkpoints/v1/llava-v1.7-7b')
model_name = get_model_name_from_path(model_path)
_, v1_model, _, _ = load_pretrained_model(model_path, model_base, model_name)

