from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import requests

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/Eiffel_tower-Paris.jpg/960px-Eiffel_tower-Paris.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)
print("Caption:", caption)
