import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache, Gemma3ForCausalLM, GemmaTokenizerFast, AutoProcessor, Gemma3nForCausalLM

PATH = "C:/Users/user/LLM/gemma3"
quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

model = Gemma3ForCausalLM.from_pretrained(PATH,
                                        device_map="auto",
                                        quantization_config=quantization_config,
                                        torch_dtype=torch.bfloat16,
                                        low_cpu_mem_usage=True
                                        )

model = model.eval()
tokenizer = GemmaTokenizerFast.from_pretrained(PATH)

agent_instruction = """<start_of_turn>user
可用的工具 (Functions) 列表：
[
  {
    "name": "add_memory",
    "description": "從user的話語中，決定這句話是不是有記憶的需求，如果有，就啟動這個函數。從prompt中獲取重點資訊，再呼叫此函數記憶",
    "parameters": {
      "type": "object",
      "properties": {
        "_type": {
          "type": "string",
          "description": "這項資訊是什麼類別，例如: '生日'、'必做清單'等等"
        },
        "message": {
          "type": "string",
          "description": "user詢問的問題，例如: '我明天該做什麼事情'",
        }
      },
      "required": ["_type", "message"]
    }
  },
  {
    "name": "get_memory",
    "description": "啟動RAG，從過去儲存的資訊中找到回答目前所必須要的記憶資訊，提供記憶上的回應，例如: 詢問生日，你會啟動RAG，找到生日是01/01",
    "parameters": {
      "type": "object",
      "properties": {
        "message": {
          "type": "string",
          "description": "user詢問的問題，例如: '我明天該做什麼事情'",
        }
      },
      "required": ["_type", "message"]
    }
  },
  {
    "name": "no_call",
    "description": "如果沒有要啟動任何工具，則使用此函數",
    "parameters": {}
  },
]
您可以使用以上工具。如果您決定呼叫任何函式，您「必須」僅回覆以下 JSON 格式，不得包含其他任何文字：
{"name": "函式名稱", "parameters": {"參數名稱": 參數值}}
"""
def AI_Agent(prompt):
    # ----- 結果儲存 ----- #
    res = list()

    # ----- Prompt token產生 ----- #
    MSG = agent_instruction + prompt + "<end_of_turn>\n<start_of_turn>model"
    input_ids = torch.tensor(tokenizer.encode(MSG)).to(model.device)
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
            model(input_ids=chunk, use_cache=True, past_key_values=past_key_values)
            st = ed
    
    # ----- Auto Regressive生成 ----- #
    input_ids = input_ids[:, -1:]
    try:
        for _ in range(32768):
            with torch.no_grad():
                # ----- Update position ----- #
                ed += 1

                # ----- Update model kwargs ----- #
                cache_position = torch.arange(past_key_values.get_seq_length(layer_idx=0)-1, 
                                            past_key_values.get_seq_length(layer_idx=0), 
                                            dtype=torch.long, 
                                            device = model.device)
                # ----- 生成token ----- #
                outputs = model(input_ids=input_ids, use_cache=True, past_key_values=past_key_values, cache_position=cache_position)
                logits = outputs.logits
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                token_id = next_token.item()
                input_ids = next_token

                # ----- 判斷是否終止 ----- #
                if token_id in eos_token_ids:
                    break

                # ----- 紀錄token ----- #
                res += [tokenizer.decode(token_id)]
        return "".join(res)
                
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

msg = """<start_of_turn>user
[所有回應一律用繁體中文回答]{prompt}<end_of_turn>
<start_of_turn>model
"""
class AI_Assistant:
    def __init__(self):
        self.past_key_values = DynamicCache()

    def generate(self, prompt):
        # ----- 結果儲存 ----- #
        res = list()

        # ----- Prompt token產生 ----- #
        MSG = msg.format(prompt=prompt)
        input_ids = torch.tensor(tokenizer.encode(MSG)).to(model.device)
        input_ids = input_ids.unsqueeze(0)
        eos_token_ids = [tokenizer.eos_token_id, 106]

        # ----- Cache宣告 ----- #
        past_key_values = self.past_key_values
        
        # ----- Prefill ----- #
        chunks = torch.split(input_ids[:, :-1], 32, dim=-1)
        st = 0
        ed = 0
        with torch.no_grad():
            for chunk in chunks:
                ed = st + chunk.shape[1]
                model(input_ids=chunk, use_cache=True, past_key_values=past_key_values)
                st = ed
        
        # ----- Auto Regressive生成 ----- #
        input_ids = input_ids[:, -1:]
        try:
            for _ in range(32768):
                with torch.no_grad():
                    # ----- Update position ----- #
                    ed += 1

                    # ----- Update model kwargs ----- #
                    cache_position = torch.arange(past_key_values.get_seq_length(layer_idx=0)-1, 
                                                past_key_values.get_seq_length(layer_idx=0), 
                                                dtype=torch.long, 
                                                device = model.device)
                    # ----- 生成token ----- #
                    outputs = model(input_ids=input_ids, use_cache=True, past_key_values=past_key_values, cache_position=cache_position)
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
                    
                    # ----- 記憶 ----- #
                    self.past_key_values = past_key_values
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
    def reset(self):
        import gc
        del self.past_key_values
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.past_key_values = DynamicCache()