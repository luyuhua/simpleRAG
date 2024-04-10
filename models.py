from langchain.llms.base import LLM
import openai



# 自定义LLM
class OpenaiApi(LLM):
    @property
    def _llm_type(self) -> str:
        return "OpenaiApi"
    
    def _call(self,prompt,stop = None,run_manager = None,**kwargs,) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        client = openai.OpenAI(
            api_key="sk-YzvJyUYM1qBzVkbu0rAuLhiYT9tjYmDNz5vYlJRvzRCVC8Mz",
            base_url="https://api.chatanywhere.tech/v1"
        )

        messages = [{'role': 'user','content': prompt},]
        completion = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
        return completion.choices[0].message.content
    

if __name__ == '__main__':
    llm = OpenaiApi()
    print(llm.invoke('你是谁？'))