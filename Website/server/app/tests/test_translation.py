from server.app.models_ai.nllb import translation_model

text = "Hello, how are you?"

translated = translation_model.translate(
    text=text,
    source_lang="eng_Latn",
    target_lang="urd_Arab"
)

print("\nOriginal:")
print(text)

print("\nTranslated:")
print(translated)