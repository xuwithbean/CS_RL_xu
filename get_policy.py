# Please install OpenAI SDK first: `pip3 install openai`
import os
from openai import OpenAI
class policy_agent:
    def __init__(
        self,
        #api_key:str='DEEPSEEK_API_KEY',
        base_url:str="https://api.deepseek.com",
        #model:str="deepseek-chat",
    ):
        #self.api_key=api_key
        self.base_url=base_url,
        #self.model=model
        self
    def getans(self,que,choice):
        client = OpenAI(
            api_key=os.environ.get('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com")

        response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "我现在的状态是："+que+"，请你从以下选项选择我下一步应该怎么做"+choice+"只回答选项的大写英文字母"},
        ],
        stream=False
        )
        return response.choices[0].message.content