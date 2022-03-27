from transformers import MarianMTModel, MarianTokenizer

src_text = [
    ">>fr<< this is a sentence in english that we want to translate to french",
    ">>pt_BR<< This should go to portuguese",
    ">>es<< And this to Spanish",
    ">>en<< O Bolsonaro é um péssimo presidente e o Lula vai ganhar no primeiro turno",
]

model_name = "Helsinki-NLP/opus-mt-en-ROMANCE"

tokenizer = MarianTokenizer.from_pretrained(model_name)

print(tokenizer.supported_language_codes)

model = MarianMTModel.from_pretrained(model_name)

translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))

tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]


from huggingface_hub import list_models

model_list = list_models()
org = "Helsinki-NLP"
model_ids = [x.modelId for x in model_list if x.modelId.startswith(org)]
suffix = [x.split("/")[1] for x in model_ids]
old_style_multi_models = [f"{org}/{s}" for s in suffix if s != s.lower()]



from transformers import MarianMTModel, MarianTokenizer
from typing import Sequence

class Translator:
    def __init__(self, source_lang: str, dest_lang: str) -> None:
        self.model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{dest_lang}'
        self.model = MarianMTModel.from_pretrained(self.model_name)
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        
    def translate(self, texts: Sequence[str]) -> Sequence[str]:
        tokens = self.tokenizer(list(texts), return_tensors="pt", padding=True)
        translate_tokens = self.model.generate(**tokens)
        return [self.tokenizer.decode(t, skip_special_tokens=True) for t in translate_tokens]
        
marian_ru_en = Translator('pt', 'ca')
marian_ru_en.translate(['como vai voce'])
# Returns: ['That being too conscious is a disease, a real, complete disease.']




from transformers import pipeline
classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

results = classifier(["Lula é um ótimo presidente", "Bolsonaro é um péssimo presidente"])

for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")



