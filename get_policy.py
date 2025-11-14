from transformers import AutoModelForCausalLM, AutoTokenizer
class policy_agent:
    def __init__(
        self,
        model_name: str = "/home/xu/code/CS_RL_xu/model/Qwen3-0.6B",
    ):
        self.model_name = model_name
    def getans(self,strs):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        prompt = strs
        choice = "A.前进,B.向后转,C.向右转,D.向左转,E.开火"
        messages = [
            {"role": "user", "content": "现在的情况是"+prompt+"请你从如下选项中选择我应该进行的动作："+choice+"请仅输出选项大写英文字符"}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=32768
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        #thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        return content
        