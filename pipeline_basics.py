from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


texts= [
    "I love using Hugging face", 
    "This is terrible", 
    "You did an amazing job",
    "The irreristible urge to subvert all expectations",
    "Sometimes I know good things fall apart, but maybe your confidence wont"
]

for text in texts: 
    result= classifier(text)[0]
    label= result['label']
    confidence = result["score"]

    print("\n" + "-" * 20 + "\n")
    print(f"Text: {text}")
    print(f"Sentiment: {label}, Confidence: {confidence:.3f}")