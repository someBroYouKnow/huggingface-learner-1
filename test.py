from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("Hugging face demo Hello World!")
print(result)