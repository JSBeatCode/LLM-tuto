import tiktoken

models_to_check = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "gpt-4"]

for m in models_to_check:
    enc = tiktoken.encoding_for_model(m)
    print(f"{m:<20}")
    print(enc)
    # print(f"  {m:<20} → Encoding: {enc.name}")
