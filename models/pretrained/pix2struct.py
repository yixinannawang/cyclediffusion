from transformers import AutoProcessor, Pix2StructForConditionalGeneration

# Initialize processor and model
processor = AutoProcessor.from_pretrained("google/pix2struct-textcaps-base")
model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-textcaps-base")
