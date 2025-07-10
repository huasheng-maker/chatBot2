import configparser
from llm_api import LLMAPIFactory
from prompt_manager import PromptManager
from response_generator import ResponseGenerator

# 读取配置文件
config = configparser.ConfigParser()
config.read('config.ini')

# 获取配置信息
deepseek_api_key = config.get('deepseek', 'api_key')
deepseek_base_url = config.get('deepseek', 'api_base', fallback="https://api.deepseek.com/v1")

max_tokens = int(config.get('chat', 'max_tokens'))
temperature = float(config.get('chat', 'temperature'))
model = config.get('chat', 'model')
# 创建LLM API实例
llm_api = LLMAPIFactory.create_api(
    "deepseek", 
    api_key=deepseek_api_key,
    base_url=deepseek_base_url
)


# 创建提示词管理器
prompt_manager = PromptManager()

# 创建回复生成器
response_generator = ResponseGenerator(llm_api, prompt_manager)

def group_chat():
    print("欢迎进入智能体群聊！输入 '退出' 结束聊天。")
    while True:
        user_input = input("你: ")
        if user_input == "退出":
            break
        response = response_generator.generate_response(user_input, temperature=temperature, max_tokens=max_tokens)
        print("智能体: ", response)

if __name__ == "__main__":
    group_chat()