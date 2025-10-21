import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HybridCache, Gemma3ForCausalLM, GemmaTokenizerFast, DynamicCache

# ----- 模型載入 ----- #
PATH = "D:\LLM\gemma\gemma3_4b"
quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
llm_model = Gemma3ForCausalLM.from_pretrained(
    PATH,
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True
    )
llm_model = llm_model.eval()
tokenizer = GemmaTokenizerFast.from_pretrained(PATH)

# ----- 回應函式 ----- #
msg = """<start_of_turn>user
[所有回應一律用繁體中文回答]{prompt}<end_of_turn>
<start_of_turn>model
"""

def predict(params):
    # ----- 參數讀取 ----- #
    prompt = params.get("prompt", "")

    # ----- 結果儲存 ----- #
    res = list()

    # ----- Prompt token產生 ----- #
    MSG = msg.format(prompt=prompt)
    input_ids = torch.tensor(tokenizer.encode(MSG)).to(llm_model.device)
    input_ids = input_ids.unsqueeze(0)
    eos_token_ids = [tokenizer.eos_token_id, 106]

    # ----- Cache宣告 ----- #
    past_key_values = DynamicCache()
    
    # ----- Prefill ----- #
    chunks = torch.split(input_ids[:, :-1], 32, dim=-1)
    st = 0
    ed = 0
    with torch.no_grad():
        for chunk in chunks:
            ed = st + chunk.shape[1]
            llm_model(input_ids=chunk, use_cache=True, past_key_values=past_key_values)
            st = ed
    
    # ----- Auto Regressive生成 ----- #
    input_ids = input_ids[:, -1:]
    try:
        for _ in range(32768):
            with torch.no_grad():
                # ----- Update position ----- #
                ed += 1

                # ----- Update model kwargs ----- #
                # cache_position = torch.arange(ed-1, ed, dtype=torch.long, device = llm_model.device)
                cache_position = torch.arange(past_key_values.get_seq_length(layer_idx=0)-1, 
                                              past_key_values.get_seq_length(layer_idx=0), 
                                              dtype=torch.long, 
                                              device = llm_model.device)
                # ----- 生成token ----- #
                outputs = llm_model(input_ids=input_ids, use_cache=True, past_key_values=past_key_values, cache_position=cache_position)
                logits = outputs.logits
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                token_id = next_token.item()
                input_ids = next_token

                # ----- 判斷是否終止 ----- #
                if token_id in eos_token_ids:
                    break

                # ----- 紀錄token ----- #
                res += [tokenizer.decode(token_id)]
                yield res[-1]
                
                # ----- 輸出文字字串 ----- #
                # print(res[-1], end="", flush=True)
    except KeyboardInterrupt as e:
        pass
    finally:
        for item in ("input_ids", "outputs", "ogits", "next_token", "token_id"):
            try:
                eval(f"del {item}")
            except:
                pass
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        