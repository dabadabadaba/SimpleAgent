from openai import OpenAI
from pydantic import BaseModel, SecretStr
from pathlib import Path
import json, os
class Independent_LLM_Tools:
    def __init__(self, llm_config, timeout = 60):
        self.BASE_URL = llm_config.get('base_url', '')
        self.API_KEY = os.getenv(llm_config.get('api_key', ''))
        self.MODEL = llm_config.get('model', None)
        self.timeout = timeout
    def llm_request(self, system_prompt, user_prompt, resp_format = 'json_object'):
        client = OpenAI(
            base_url=self.BASE_URL,
            api_key=self.API_KEY,
            timeout = self.timeout
        )

        response = client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            # 3. 加入 response_format 参数，开启 JSON 结构化输出
            response_format={
                'type': resp_format
            },
            extra_body={"thinking": {"type": "disabled"}}
        )
        return response.choices[0].message.content
    def llm_file_map(self, context):
        system_prompt = """
            你是一个车辆工程专家和技术文档专家。
            """
        user_prompt = f'''
    # Task
    首先判断文本是否是代码，以及哪种代码，并严格按照以下规则进行结构化总结，最终仅返回 Markdown 格式的结果,禁止输出```Markdown```这类块标记：
    # Rules
    - **若为代码**：深度解析代码结构。提取所有定义的“类（Class）”与“函数/方法（Function）”。针对每一项，必须清晰列出：
        - 类和函数的拓扑关系
        - 功能描述
        - 接口定义*包含输入参数、返回值*
    - **若为非代码**：提炼文本核心要点，概括全文，精简字数。
    # Constraints
    - 无需解释分析过程。
    - 控制字数，保持简短
    
    # 输入文本：
    {context}
        '''
        content = self.llm_request(system_prompt, user_prompt, "text")
        return content
    def llm_excel_map(self, context):
        system_prompt = """
            你是一个车辆工程专家和技术文档专家。
            """
        user_prompt = f'''
    # Task
    将excel原始数据归纳总结，哪些是表头，哪些是补充信息，哪些是数据，有什么特点和规律。
    # Rules
    - 在 Excel 中，这类位于前几行或前几列、用于说明数据含义或类型的文字为表头。
    - 表头是理解表格的关键，要保留全部表头信息。
    - 对于数据的总结要包含例子，方便理解数据的类型和含义。
    - 要体现表的结构。
    - 控制字数，简明扼要，最终仅返回 Markdown 格式的结果,禁止输出```Markdown```这类块标记。
    # 输入文本：
    {context}
        '''
        content = self.llm_request(system_prompt, user_prompt, "text")
        return content
    def llm_response_rectify(self, context):
        system_prompt = """
            你是一个车辆工程专家。
            """
        user_prompt = f'''
    # Task
    判断原始信息是否发生了错乱，例如出现了大量重复的段落，陷入了循环，无法收敛。
    如果出现错乱，仅收集其中少量有意义的信息，抛弃错乱的信息。
    # 原始文本：
    {context}
        '''
        content = self.llm_request(system_prompt, user_prompt, "text")
        return content

    def llm_context_rectify(self, system_prompt, context):  # 用于最后纠正langgragh的行为
        system_prompt = system_prompt
        user_prompt = f'''
    # 目标文本：
    {context}
        '''
        content = self.llm_request(system_prompt, user_prompt, "text")
        return content
    def llm_json_rectify(self,system_prompt, context): # 强制输出json，用于最后纠正Router的行为
        user_prompt = f'''
    # 目标文本：
    {context}
        '''
        content = self.llm_request(system_prompt, user_prompt, "json_object")
        return content
    def llm_messages_compress(self, context):
        system_prompt = """
            你是一个车辆工程专家和技术文档专家。
            """
        user_prompt = f'''
    你需要对下面历史对话做总结，严格遵守规则：
    1. 先总结：任务目标、已排查的问题、确认根因、已执行操作、关键约束；
    2. 再简短综合总结之前做过什么；
    3. 剔除无意义复述、重复验证话术，禁止啰嗦与复读；
    4. 输出为一段通顺文本，长度控制在250~450字；
    5. 不要续写推理，只做客观汇总；
    6. 注意来自Human的指令，这部分信息要保留并标记为Human的指令。
    7. 返回 Markdown 格式的结果,禁止输出```Markdown```这类块标记。
    # 历史对话文本：
    {context}
        '''
        content = self.llm_request(system_prompt, user_prompt, "text")
        return content

    def llm_identify_tc(self, context): # 识别出测试用例
        system_prompt = """
            你是一个车辆技术工程专家。
            """
        text4segment = f'''
        # 以下汽车工程相关的测试文本信息：
        {context}
        以上信息是否是对测试用例的描述（测试需求，测试目的，测试步骤等），
        首先确定以上输入信息中包含几个测试用例。
        如果包含一个或多个，将其按照测试用例识别，然后其按照语义规律分割组合为Dict，其中的表格或列表转为markdown，*保留原文所有信息*
        如果不包含，返回空的Dict。
        EXAMPLE JSON OUTPUT:
        {{
        "测试用例名”:["测试需求A","测试步骤1","测试步骤2","测试需求B。","测试步骤1"],
        "TC_EngineStart_001“:["1.ControlDeck手动上电模式启动. 以时间为x轴，转速为Y轴，记录启动过程。","2.若启动后转速大于等于700转，且喷油量大于0，则判断启动成功"]
        }}
        '''
        content = self.llm_request(system_prompt, text4segment)
        return json.loads(content)
    def llm_segmente_context(self, context): #把未知形式的文本改造成分段成List
        system_prompt = """
        你是一个车辆工程专家。
        """
        text4segment = f'''
    # 以下汽车工程相关的测试文本信息：
    {context}
    以上信息是一些测试相关描述，现需要将其按照语义规律分割成若干段，识别其中的表格或列表转为markdown，*尽量保留原文信息，不要丢失*只返回分割后的结果
    EXAMPLE JSON OUTPUT:
    [
         "ECU复位：复位开关先置1后置0，转速扭矩控制模式置0，转速控制模式置0。",
         "启动测试：ControlDeck手动上电模式启动. 以时间为x轴，转速为Y轴，记录启动过程。若启动后转速大于等于700转，且喷油量大于0，则判断启动成功"
    ]
    '''
        content = self.llm_request(system_prompt,text4segment)
        return json.loads(content)

    def llm_confirm_mapping(self, mapping_markdown, context):
        # 转化为 Markdown 表格字符串
        system_prompt = """
        你是一个车辆工程专家。从以下汽车工程相关的测试文本，筛选出涉及的信号。
        """
        text4mapping = f'''
    # 测试文本信息：
    {context}
    # 信号列表：
    {mapping_markdown}
    要求采用信号列表中已有的信号，如果信号列表中有匹配的信号，以信号列表为准。注意参考description的补充信息。
    如果信号列表中没有匹配的信号，则创建新的信号名并直接返回JSON形式Dict(信号名为KEY，Keywords为描述性关键词)。
    信号命名遵循python语法，
    如果文本中不包含明确的信号信息，允许返回空Dict。
    另外识别文本中涉及到的文件。
    EXAMPLE JSON OUTPUT:
    {{
         “matched_signals”: [['signalLabel(string)', 'writable(bool)', 'description(string)']...],
         “created_signals”: [['signalLabel(string)', 'writable(bool)', 'description(string)']...],
         “related_files": [['fileName(string)','fileType(string)','filePath(string)']...]
    }}
    # 参考以上信息列表，从以下测试文本中提取涉及的信号：
    {context}
    选择赋值信号时注意信号必须是writable。输出务必精简，不要冗余，控制长度。
    '''
        content = self.llm_request(system_prompt,text4mapping)
        return json.loads(content)

    def generate_allura_step(self, stepMapping, context, stepExamples, pre_steps):

        system_prompt = """
    你是一个测试专家。请根据用户的需求基于pytest和allura生成python测试代码。
    要求：
    1. 代码注释要详细，包含测试文本信息
    2. 只输出步骤的代码和详细的注释，以with allure.step()开始,参考示例
    3. 代码用```python```代码块包裹
    """
        text4steps = f'''
    以下汽车工程相关的测试文本信息提取测试步骤。
    # 测试文本信息：
    {context}
    # 参考例子：
    {stepExamples}
    # 前置步骤(前置步骤与当前要转化的步骤存在因果关系)：
    {pre_steps}
    # 要求采用以下信号：
    {stepMapping}
    转化以下测试文本信息为pytest-allura测试步骤(如果缺少必要信号，在注释中注明）：
    {context}
    '''
        content = self.llm_request(system_prompt, text4steps, "text")
        return content

    def llm_pytest_TC(self, generated_steps, reference = '无'):
        system_prompt = """
    你是一名测试专家，请基于 pytest + allure 生成可运行的 Python 自动化测试代码，严格遵守以下要求：
    1. 输出标准 pytest + allure 规范测试代码；
    2. 仅输出测试相关代码与详细注释，注释还原测试描述原文，代码完整、直接可执行；
    3. 代码追求简洁高效，尽量避免额外自定义函数；如必须新增自定义函数，需要在注释清晰标注函数输入参数、返回值定义；
    4. 返回纯净Python代码，禁止输出```python、```这类代码块标记, 不输出代码以外多余描述、解释文本。
    """
        text4steps = f'''
    以下汽车工程相关的测试文本信息提取测试步骤。
    # 参考信息：
    {reference}
    # 测试步骤信息：
    {generated_steps}
    将以上信息是一个测试用例的各个步骤。
    先检查以上步骤的代码是否有可以省略的冗余，优化代码。
    遵循以下框架：
    try:
        1. 写入和采集，除非有明确的退出测试条件，否则不做判断
        2. 绘制表格和图片，展示测试数据
        3. 在最后集中进行数据的判断，
    except:
        搜集打印报错信息
    finally:
        确保一些reset步骤比如复位参数和下电被执行，以便连续执行测试
    整合为完整可执行的*一个测试用例*
    '''
        content = self.llm_request(system_prompt, text4steps, "text")
        print(system_prompt)
        return content



if __name__ == "__main__":
    pass
