import torch
import open_clip
from PIL import Image
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.nn as nn
# Step 1: Load CLIP Model and LLaMA (used instead of GPT-2)
device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
clip_model = clip_model.to(device)
model_name = "gpt2"
# 加载GPT-2
gpt2_model = GPT2LMHeadModel.from_pretrained(model_name)
gpt2_tokenizer =  GPT2Tokenizer.from_pretrained(model_name)

class ProjectionLayer(nn.Module):
    def __init__(self, clip_dim, gpt2_dim):
        super().__init__()
        self.projection = nn.Linear(clip_dim, gpt2_dim)
    
    def forward(self, x):
        return self.projection(x)

# CLIP ViT-B/32 输出维度是512，GPT-2的嵌入维度是768
projection_layer = ProjectionLayer(512, 768).to(device)
class CombinedModel(nn.Module):
    def __init__(self, clip_model, gpt2_model, projection_layer):
        super().__init__()
        self.clip_model = clip_model
        self.gpt2_model = gpt2_model
        self.projection_layer = projection_layer
    
    def forward(self, image, input_ids, attention_mask=None):
        # 获取图像特征
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
        
        # 投影图像特征
        projected_image_features = self.projection_layer(image_features)
        
        # 获取GPT-2的嵌入
        gpt2_embeds = self.gpt2_model.transformer.wte(input_ids)
        
        # 将图像特征与文本嵌入结合
        combined_embeds = torch.cat([projected_image_features.unsqueeze(1), gpt2_embeds], dim=1)
        
        # 运行GPT-2
        outputs = self.gpt2_model(inputs_embeds=combined_embeds, attention_mask=attention_mask)
        
        return outputs

combined_model = CombinedModel(clip_model, gpt2_model, projection_layer).to(device)

def generate_description(model, image_path, prompt="This image shows", max_length=50):
    # 预处理图像
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    
    # 编码提示文本
    input_ids = gpt2_tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # 生成文本
    with torch.no_grad():
        output_sequences = model.gpt2_model.generate(
            input_ids=input_ids,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
    
    # 解码生成的文本
    generated_text = gpt2_tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    return generated_text


# 使用示例
image_path = 'D:\\学习\\研究生\\顶会\\熊猫.jpg'
generated_description = generate_description(combined_model, image_path)
print(generated_description)