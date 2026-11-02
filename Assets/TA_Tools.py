from langchain_core.tools import tool
from DS_Utilities import mapping_list_markdown, extract_excel_with_pandas, multi_file_read
import json
from pathlib import Path
# ================== 配置类 ==================
class LLM_CONFIG: #当tool本身要用到llm，添加此配置信息
    llm = None

pytest_code_reference = """
## 一下代码均为语法和套路示例，其中涉及到的变量名和文件名不具有参考价值
import logging
import platform
import sys
import time
import traceback
from . import utils
import pytest
import allure
from dspace.testautomation import Capturing
from dspace.testautomation.testing.testenv import TestEnvironmentAccess
import matplotlib.pyplot as plt
import io

def plot_xy_curve(x_list: list, y_list: list, x_label="转速(rpm)", y_label="扭矩(Nm)", title="外特性曲线"):
    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_list, y_list, marker='o', linestyle='-', color='#2E86AB', label="实测扭矩")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    # 内存缓存图片，不生成本地文件
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    plt.close(fig)  # 释放内存
    # 嵌入allure报告
    allure.attach(
        buf.read(),
        name=title,
        attachment_type=allure.attachment_type.PNG
    )
    buf.close()

def test_Engine_Torque(ta: TestEnvironmentAccess):
    try:
        #ECU复位. 
        #ControlDeck复位开关先置1后置0，转速扭矩控制模式置0，转速控制模式置0
        with allure.step("ECU复位. ControlDeck复位开关先置1后置0,转速扭矩控制模式置0,转速控制模式置0"):
            ta.vars.RESET.value = 1
            time.sleep(0.2)
            ta.vars.RESET.value = 0
            ta.vars.Torque_Controller_Mode.value = 0
            ta.vars.Speed_Controller_Mode.value = 0
            
        #ControlDeck上30电，手动上电模式置1
        with allure.step("ControlDeck上30电,手动上电模式置1"):
            ta.vars.KL30.value = 1
            time.sleep(0.1) 
        # 启动测试
        with allure.step("启动测试.\n ControlDeck手动上电模式启动.\n 以时间为x轴，转速为Y轴，记录启动过程。若启动后转速大于等于700转，且喷油量大于0，则判断启动成功"):
            with Capturing(
                        [
                            ta.vars.Engine_Speed,
                        ],
                    ) as capture:
                # ControlDeck手动上电模式启动
                ta.vars.Manuel_Control.value = 1
                # 以时间为x轴，转速为Y轴，记录启动过程。若启动后转速大于等于700转，且喷油量大于0，则判断启动成功。timeout为15s
                ta.vars.State_StartButton.value = 2 #按下启动键
                for i in range(16):
                    time.sleep(1)
                    oil_inject_quantity = ta.vars.q_Mean_Inj.value
                    engine_speed = ta.vars.Engine_Speed.value
                    if engine_speed >= 700 and oil_inject_quantity > 0:
                        break
            data_frame = capture.get_data_frame()
            utils.plot_dataframe_to_report(
                data_frame,
                layout=[
                ["Engine_Speed"],
                ],
            )
            assert i < 15, "启动超时"
        
        # 外特性测试
        with allure.step(
            '''外特性测试 
            ControlDeck转速扭矩控制模式，转速1800rpm,扭矩200Nm,运行60S，扭矩为0，关闭转速扭矩模式，改为纯转速模式。
            油门开度设定为100%
            根据EXCEL表格 要求记录数据（第一列为目标转速，第二列为目标扭矩，第三列为实际测量扭矩），每个点每1秒记录一次数据，记录10次，取平均值记录到表格第三列中，平均值保留2位小数。实际扭矩与目标扭矩计算差值，差值小于2NM，则测试通过
            以转速为X轴，实际测量扭矩为Y轴绘制外特性曲线，以及输出转速扭矩表。
            '''):
            # ControlDeck转速扭矩控制模式，转速1800rpm,扭矩200Nm,运行60S，扭矩为0，关闭转速扭矩模式，改为纯转速模式。目的：热机。
            ta.vars.Torque_Controller_Mode.value = 1 #使能扭矩控制
            ta.vars.Speed_Controller_Mode.value = 1 #使能转速控制
            ta.vars.Trq_Engine_Set.value = 200 #写入扭矩
            ta.vars.Engine_Speed_Set.value = 1800 #写入转速
            time.sleep(60)
            ta.vars.Trq_Engine_Set.value = 0
            ta.vars.Torque_Controller_Mode.value = 0
            # 油门开度设定为100%
            ta.vars.AccPedal_Pos.value = 100
            # 根据EXCEL表格（D:\example_folder\example_map.xlsx）要求记录数据（第一列为目标转速，第二列为目标扭矩，第三列为实际测量扭矩），每个点每1秒记录一次数据，记录10次，取平均值记录到表格第三列中，平均值保留2位小数。实际扭矩与目标扭矩计算差值，差值小于2NM，则测试通过
            import pandas as pd
            actual_torque_list = []
            df = pd.read_excel(r"D:\example_folder\example_map.xlsx", usecols=[0, 1], header=None, skiprows=2).dropna()
            target_speed_list = df.iloc[:, 0].tolist()
            target_torque_list = df.iloc[:, 1].tolist()
            assert len(target_speed_list) == len(target_torque_list), "数据长度不匹配"
            def get_torque_with_speed(speed:float): #每个点每1秒记录一次数据，记录10次，取平均值        
                for i in range(10):
                    time.sleep(1)
                    torque_samplings.append(ta.vars.Trq_MeanEff_Engine.value)
                return sum(torque_samplings) / len(torque_samplings)
            flag = True
            msg = ''
            for i in range(len(target_speed_list)):
                speed = target_speed_list[i] #获得新速度
                target_torque = target_torque_list[i] #获得目标扭矩
                with allure.step(f"Loop{i}, Target Speed: {target_speed_list[i]}rpm, Expected Torque: {target_torque_list[i]}Nm"):
                    ta.vars.Engine_Speed_Set.value = speed
                    time.sleep(5) # 赋值转速后，让扭矩重新稳定
                    with Capturing(
                            [
                                ta.vars.Trq_MeanEff_Engine,
                                ta.vars.Engine_Speed,
                            ],
                        ) as capture:
                        #采样十次取平均值获得实际扭矩
                        torque_samplings = []
                        for i in range(10):
                            time.sleep(1)
                            torque_samplings.append(ta.vars.Trq_MeanEff_Engine.value)
                        actual_torque = sum(torque_samplings) / len(torque_samplings)
                    actual_torque_list.append(actual_torque)
                    data_frame = capture.get_data_frame()
                    utils.plot_dataframe_to_report(
                        data_frame,
                        name = f"Loop{i}, Target Speed: {target_speed_list[i]}rpm, Expected Torque: {target_torque_list[i]}Nm",
                        layout=[
                            ["Trq_MeanEff_Engine"], ["Engine_Speed"],
                        ],
                    )
                    torque_diff = abs(actual_torque - target_torque)
                    if torque_diff > 2:
                        flag = False
                        msg += r'\n' + f"扭矩偏差超限！转速{speed}rpm 目标{target_torque}Nm，实际{actual_torque:.2f}Nm，差值{torque_diff:.2f}Nm，允许±2Nm"
            # 以转速为X轴，实际测量扭矩为Y轴绘制外特性曲线，以及输出转速扭矩表。
            df = pd.DataFrame(zip(target_speed_list, target_torque_list, actual_torque_list), columns=["目标转速", "目标扭矩", "实际扭矩"])
            table = df.to_html(index=False)
            allure.attach(
                table, 
                name="测试结果数据表", 
                attachment_type=allure.attachment_type.HTML)
            #画特性曲线图，画图避免使用汉字，会造成乱码
            plot_xy_curve(target_speed_list, actual_torque_list, x_label="speed(rpm)", y_label="torque(Nm)", title="External characteristic curve")
            assert flag, msg # assert后置
    except Exception as e:
        # 获取完整的报错信息字符串（包含 Traceback 和异常类型）
        full_error_msg = traceback.format_exc()    
        assert False, full_error_msg
    finally:
        # 油门开度等于0，转速等于0，T15等于0，等待60S，T30等于0. 目的：停机
        with allure.step( "油门开度等于0，转速等于0，T15等于0，等待60S，T30等于0."):
            ta.vars.AccPedal_Pos.value = 0
            ta.vars.Trq_Engine_Set.value = 0
            ta.vars.Engine_Speed_Set.value = 0
            ta.vars.Torque_Controller_Mode.value = 0
            ta.vars.Speed_Controller_Mode.value = 0
        #下电
        with allure.step("下电"):
            ta.vars.State_StartButton.value = -1 
            time.sleep(60)
            ta.vars.KL30.value = 0
参考以上信息生成测试代码！
"""
def handle_test_context(file_path):
    md = multi_file_read(file_path)
    test_context_dict = LLM_CONFIG.llm.llm_identify_tc(md)
    if test_context_dict == {}:
        return {}, f"{file_path}未发现测试用例\n"
    with open(file_path+'.json', "w", encoding="utf-8") as f:
        json.dump(test_context_dict, f, ensure_ascii=False, indent=4)
    print(f"生成测试用例json文件 {file_path+'.json'}")
    return test_context_dict, f"生成测试用例json文件 {file_path+'.json'}\n"
@tool
def mapping_analysis_report(mapping_path, tc_context_file): #分析mapping相对于当前测试所缺失的信息，并产生分析报告
    """
    分析mapping和测试描述文本，搜集相对于当前测试所缺失的信息，并产生分析报告
    Args:
        mapping_path: mapping csv文件路径 **绝对路径**
        tc_context_file: 包含测试步骤的测试文本文件路径
    Return:
        分析报告pre_generation_report的文本
    """
    reportList = []
    stepList = []
    try:
        mapping_table = mapping_list_markdown(
            extract_excel_with_pandas(mapping_path, start_row=2, columns=[1, 5, 6]),
            ['Label', 'IsWriable', 'Description'])
    except Exception as e:
        return str(e) + "\n请输入绝对路径\n"
    try:
        report = 'Mapping分析结果如下：\n'
        original_context_Dict, msg = handle_test_context(tc_context_file)
        report += msg
        if original_context_Dict == {}:
            return report
        for tc in original_context_Dict:
            report += f"# 测试用例：{tc}"
            for context in original_context_Dict[tc]:
                steps_mapping = LLM_CONFIG.llm.llm_confirm_mapping(mapping_table, context)
                stepList.append([context, steps_mapping])
                if len(steps_mapping["created_signals"]) > 0:
                    reportList.append([context, steps_mapping])
            if len(reportList) == 0:
                report += "Mapping已分析，无异议\n"
            else:
                for step in  stepList:
                    report += f"### {step[0]}\n"
                    report += "已匹配的变量:\n"
                    report += f"{step[1]['matched_signals']}\n"
                    if len(step[1]['created_signals']) > 0 :
                        report += "异议! 需要补充的变量:\n"
                        report += f"{step[1]['created_signals']}\n"
        # 获取文件夹路径（父目录）
        folder_path = Path(tc_context_file).parent
        report_file_path = folder_path / 'pre_generation_report.md'
        report_file_path.parent.mkdir(parents=True, exist_ok=True)
        report_file_path.write_text(report, encoding='utf-8')
        report += f'\n**以上内容已写入文件: {report_file_path}**\n'
        return report
    except Exception as e:
        return str(e)

@tool
def generate_pytest_case(test_case_context):
    """
        生成符合要求的pytest测试用例
        Args:
            test_case_context: 测试用例描述文本，应从pre_generation_report获取
        Return: pytest测试代码
        """
    try:
        TC_code = LLM_CONFIG.llm.llm_pytest_TC(test_case_context,reference= pytest_code_reference)
        print('generate_pytest_case\n' + TC_code)
        return "# 生成测试用例代码：\n" + TC_code + "\n# 已上为生成的代码文本"
    except Exception as e:
        return str(e)
if __name__ == "__main__":
    # 自测示例（替换为本地实际路径后运行）
    pass