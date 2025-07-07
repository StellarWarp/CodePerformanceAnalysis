import asyncio
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Annotated, List, Dict, Any, TypedDict, Optional, Literal, AsyncIterator
from dataclasses import dataclass, field

from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END, add_messages
from langgraph.types import Command, interrupt, Interrupt

json_converter_prompt = """# 角色与目标 (ROLE & GOAL)
你是一位专业的、高精度的格式转换专家。你唯一的任务，就是根据下面提供的规范，将自然语言描述的计划转换为严格的JSON格式。你必须分析给出的对话历史，在其中找到用户最终批准的计划版本，并执行转换。

# 上下文 (CONTEXT)
你收到的对话历史包含了一段关于创建性能分析工作流的完整讨论。当前的这条指令是整个对话历史的最后一条消息。最终确认的计划，就是在这条指令之前、最后一次出现的、以自然语言格式呈现的、包含多个步骤的详细工作流。

# 任务 (TASK)
1.  **定位 (IDENTIFY)**: 从对话历史的末尾向前扫描，定位最终的、完整的、包含多个步骤的计划。该计划的格式为一系列步骤，每个步骤都以“步骤 X: [任务标识符]”开头。
2.  **转换 (CONVERT)**: 根据下述严格的规范，将定位到的计划转换为一个JSON对象。
3.  **输出 (OUTPUT)**: 你的输出必须且只能是纯粹、有效的JSON对象，绝不能包含任何额外的文字、解释或像 ```json ... ``` 这样的Markdown标记。

# JSON规范与字段映射规则 (JSON SCHEMA & FIELD MAPPING)
输出必须是一个单一的JSON对象，其根键为“plan”，值是一个由任务对象组成的数组。请严格遵守以下的字段名称和映射规则：

## 根结构
{
  "plan": [ /* 任务对象数组 */ ]
}

## 任务对象结构
{
  "task_name": "string",
  "description": "string",
  "dependencies": ["string"],
  "suggested_tools": ["string"],
  "key_parameters": { "key": "value" }
}

## 字段映射规则
- **"task_name"**: 从 `步骤 X: [任务标识符]` 这一行中，提取 `[任务标识符]` 字符串。
- **"description"**: 使用 `任务描述:` 字段的完整文本内容。
- **"dependencies"**: 从 `前置依赖:` 字段中提取标识符。如果内容为“无”，则必须使用一个空数组 `[]`。如果列出了多个依赖，则创建一个包含所有依赖的字符串数组。
- **"suggested_tools"**: 从 `建议工具:` 字段中提取所有工具名称，并将它们放入一个字符串数组。
- **"key_parameters"**: 将 `关键参数:` 下方列出的所有键值对，转换为一个JSON对象。请确保所有值的类型正确（如数字、布尔值或字符串）。

现在开始转换。"""


# --- Configuration ---
@dataclass
class AgentConfig:
    """代理配置类"""
    model_name: str = "deepseek-chat"
    api_key: str = ""  # 从环境变量或配置文件读取
    base_url: str = "https://api.deepseek.com/v1"
    mcp_servers: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "Unreal insight Call Tree": {
            'url': "http://localhost:8000/sse/",
            "transport": "sse"
        }
    })
    thread_id: str = "human-in-the-loop-thread"

    # 文档路径配置
    experience_doc_path: str = "doc/experience_doc"
    system_prompt_path: str = "doc/system_prompt"
    user_prompt_path: str = "doc/user_prompt"
    utrace_file_path: str = r'C:\Users\lyq\Desktop\Work\CodePerformanceAnalysis\data\CSV\Test\20250626_215834.utrace'


# --- State Definition ---
class PlanningState(TypedDict):
    """代理状态定义"""
    messages: Annotated[List[AnyMessage], add_messages]
    plan_approved: bool
    revision_count: int
    current_plan: Optional[str]
    final_json: Optional[str]


# --- Document Manager ---
class DocumentManager:
    """文档管理器"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def read_file_safely(self, file_path: str, default_content: str = "") -> str:
        """安全读取文件内容"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            self.logger.warning(f"文件 '{file_path}' 未找到，使用默认内容")
            return default_content
        except Exception as e:
            self.logger.error(f"读取文件 '{file_path}' 时发生错误: {e}")
            return default_content

    def read_experience_doc(self) -> str:
        """读取经验文档"""
        return self.read_file_safely(self.config.experience_doc_path)

    def read_system_prompt(self) -> str:
        """读取系统提示"""
        system_prompt = self.read_file_safely(
            self.config.system_prompt_path,
            ""
        )
        return system_prompt.replace("{experience_docs}", self.read_experience_doc())

    def read_user_prompt(self) -> str:
        """读取用户提示"""
        user_prompt = self.read_file_safely(
            self.config.user_prompt_path,
            ""
        )
        return user_prompt.format(utrace_file=self.config.utrace_file_path)


# --- Streaming Helper Functions ---
def print_stream_chunk(chunk: str, end: str = ""):
    """打印流式输出块"""
    print(chunk, end=end, flush=True)


async def stream_llm_response(llm, messages: List[AnyMessage], prefix: str = "") -> str:
    """流式调用LLM并返回完整响应"""
    if prefix:
        print(f"\n{prefix}")

    full_response = ""

    try:
        # 使用astream进行流式调用
        async for chunk in llm.astream(messages):
            if hasattr(chunk, 'content') and chunk.content:
                content = chunk.content
                print_stream_chunk(content)
                full_response += content
    except Exception as e:
        error_msg = f"流式调用LLM时发生错误: {str(e)}"
        print_stream_chunk(error_msg)
        full_response = error_msg

    print()  # 换行
    return full_response


# --- Graph Nodes ---
class PlanningNodes:
    """规划节点类"""

    def __init__(self, llm, config: AgentConfig):
        self.llm = llm
        self.config = config
        self.logger = logging.getLogger(__name__)

    def planner_node(self, state: PlanningState) -> Dict[str, Any]:
        """规划节点：生成或修订计划"""
        revision_count = state.get('revision_count', 0)

        if revision_count == 0:
            prefix = "🤖 正在生成初始计划..."
        else:
            prefix = f"🤖 正在修订计划 (第 {revision_count} 次修订)..."

        try:
            # 检查是否有事件循环，如果没有则创建新的
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 如果事件循环正在运行，使用 run_coroutine_threadsafe
                    import concurrent.futures
                    import threading

                    def run_in_new_loop():
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            return new_loop.run_until_complete(
                                stream_llm_response(self.llm, state['messages'], prefix)
                            )
                        finally:
                            new_loop.close()

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_in_new_loop)
                        response_content = future.result()
                else:
                    response_content = loop.run_until_complete(
                        stream_llm_response(self.llm, state['messages'], prefix)
                    )
            except RuntimeError:
                # 没有事件循环，创建新的
                response_content = asyncio.run(
                    stream_llm_response(self.llm, state['messages'], prefix)
                )

            # 创建AI消息
            response = AIMessage(content=response_content)

            # 更新状态
            return {
                "messages": [response],
                "current_plan": response_content,
                "revision_count": revision_count + 1
            }
        except Exception as e:
            self.logger.error(f"规划节点执行错误: {e}")
            error_msg = AIMessage(content=f"生成计划时发生错误: {str(e)}")
            return {"messages": [error_msg]}

    def human_approval_node(self, state: PlanningState) -> Dict[str, Any]:
        """人工审核节点"""
        current_plan = state.get('current_plan', "")
        revision_count = state.get('revision_count', 0)

        interrupt_data = {
            "question": "请审核以下计划。如果满意，请输入 'approved'；否则请提供具体修改意见：",
            "current_plan": current_plan,
            "revision_count": revision_count
        }
        user_feedback = interrupt(interrupt_data)
        return {"messages": [HumanMessage(content=user_feedback)]}


    def json_converter_node(self, state: PlanningState) -> Dict[str, Any]:
        """JSON转换节点：将批准的计划转换为JSON格式"""
        json_instruction = HumanMessage(content=json_converter_prompt)

        try:
            # 准备完整的消息历史
            messages_for_conversion = state['messages'] + [json_instruction]

            # 使用流式输出进行JSON转换，处理事件循环
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures

                    def run_in_new_loop():
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            return new_loop.run_until_complete(
                                stream_llm_response(
                                    self.llm,
                                    messages_for_conversion,
                                    "🔄 正在将计划转换为JSON格式..."
                                )
                            )
                        finally:
                            new_loop.close()

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_in_new_loop)
                        response_content = future.result()
                else:
                    response_content = loop.run_until_complete(
                        stream_llm_response(
                            self.llm,
                            messages_for_conversion,
                            "🔄 正在将计划转换为JSON格式..."
                        )
                    )
            except RuntimeError:
                response_content = asyncio.run(
                    stream_llm_response(
                        self.llm,
                        messages_for_conversion,
                        "🔄 正在将计划转换为JSON格式..."
                    )
                )

            # 创建AI消息
            response = AIMessage(content=response_content)

            return {
                "messages": [response],
                "final_json": response_content,
                "plan_approved": True
            }
        except Exception as e:
            self.logger.error(f"JSON转换节点执行错误: {e}")
            error_msg = AIMessage(content=f"转换为JSON时发生错误: {str(e)}")
            return {"messages": [error_msg]}


# --- Routing Functions ---
def route_after_human_review(state: PlanningState) -> Literal["planner_node", "json_converter_node"]:
    """人工审核后的路由函数"""
    last_message = state['messages'][-1]

    if isinstance(last_message, HumanMessage):
        feedback = last_message.content.lower().strip()

        if 'approved' in feedback:
            return 'json_converter_node'
        else:
            return 'planner_node'

    # 默认返回规划节点
    return 'planner_node'


# --- Async Main Agent Class ---
class PlanningAgent:
    """主代理类"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.doc_manager = DocumentManager(config)
        self.logger = logging.getLogger(__name__)
        self.llm = None
        self.graph = None
        self.memory = MemorySaver()

    async def initialize(self):
        """初始化代理"""
        # 初始化LLM
        self.llm = ChatOpenAI(
            model=self.config.model_name,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            streaming=True,  # 启用流式输出
        )

        # 初始化MCP客户端
        try:
            client = MultiServerMCPClient(self.config.mcp_servers)
            tools = await client.get_tools()
            self.llm = self.llm.bind_tools(tools)
            self.logger.info("MCP工具绑定成功")
        except Exception as e:
            self.logger.warning(f"MCP工具绑定失败: {e}，继续使用基础LLM")

        # 构建图
        self._build_graph()

    def _build_graph(self):
        """构建LangGraph工作流"""
        nodes = PlanningNodes(self.llm, self.config)

        # 创建状态图
        workflow = StateGraph(PlanningState)

        # 添加节点
        workflow.add_node("planner_node", nodes.planner_node)
        workflow.add_node("human_approval_node", nodes.human_approval_node)
        workflow.add_node("json_converter_node", nodes.json_converter_node)

        # 设置入口点
        workflow.set_entry_point("planner_node")

        # 添加边
        workflow.add_edge("planner_node", "human_approval_node")
        workflow.add_edge("json_converter_node", END)

        # 添加条件边
        workflow.add_conditional_edges(
            "human_approval_node",
            route_after_human_review,
            {
                "planner_node": "planner_node",
                "json_converter_node": "json_converter_node"
            }
        )

        # 编译图
        self.graph = workflow.compile(checkpointer=self.memory)

    async def run(self) -> Dict[str, Any]:
        """运行代理"""
        if not self.graph:
            raise RuntimeError("代理未初始化，请先调用 initialize() 方法")

        # 准备初始输入
        system_prompt = self.doc_manager.read_system_prompt()
        user_prompt = self.doc_manager.read_user_prompt()

        config = {"configurable": {"thread_id": self.config.thread_id}}

        initial_input = {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ],
            "plan_approved": False,
            "revision_count": 0,
            "current_plan": None,
            "final_json": None
        }

        # 执行图
        try:
            response = self.graph.invoke(initial_input, config)

            # 处理人工交互循环
            while True:
                current_state = self.graph.get_state(config=config)

                if len(current_state.next) == 0:
                    self.logger.info("🎉 代理执行完成")
                    break

                # 显示当前计划供审核
                if response.get("__interrupt__"):
                    interrupt_data = response["__interrupt__"][-1]
                    self._display_plan_for_review(interrupt_data)

                    # 获取用户输入
                    human_input = input("\n请输入您的反馈: ").strip()

                    # 继续执行
                    response = self.graph.invoke(Command(resume=human_input), config=config)

            # 获取最终结果
            final_state = self.graph.get_state(config)
            return self._extract_final_result(final_state)

        except Exception as e:
            traceback.print_exc()
            self.logger.error(f"代理执行错误: {e}")
            raise

    def _display_plan_for_review(self, interrupt_data: Interrupt):
        """显示计划供审核"""
        print("\n" + "=" * 60)
        print("📋 计划审核")
        print("=" * 60)

        # 由于现在使用流式输出，当前计划已经在流式输出中显示了
        # 这里只显示额外的审核信息

        if "revision_count" in interrupt_data.value:
            print(f"修订次数: {interrupt_data.value['revision_count']}")

        print("\n" + interrupt_data.value.get("question", "请审核计划"))
        print("=" * 60)
        print("💡 提示: 输入 'approved' 批准计划，或输入具体修改意见")

    def _extract_final_result(self, final_state) -> Dict[str, Any]:
        """提取最终结果"""
        state_values = final_state.values

        # 获取最终的AI消息
        final_ai_message = None
        for msg in reversed(state_values['messages']):
            if isinstance(msg, AIMessage):
                final_ai_message = msg.content
                break

        result = {
            "final_plan": final_ai_message,
            "revision_count": state_values.get('revision_count', 0),
            "plan_approved": state_values.get('plan_approved', False),
            "final_json": state_values.get('final_json')
        }

        return result


# --- Main Function ---
async def main():
    """主函数"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 创建配置（在实际应用中，应从环境变量或配置文件读取）
    config = AgentConfig(
        api_key="sk-7b32e9e21abd4eea9c7a9db728401324",  # ❗️ 请替换为您的实际API密钥
        thread_id="planning-agent-thread-v2"
    )

    # 创建并运行代理
    agent = PlanningAgent(config)

    try:
        await agent.initialize()
        result = await agent.run()

        # 显示最终结果
        print("\n" + "=" * 60)
        print("🎉 最终结果")
        print("=" * 60)
        print(f"修订次数: {result['revision_count']}")
        print(f"计划已批准: {result['plan_approved']}")

        # 最终计划已经通过流式输出显示了，这里只显示摘要
        if result['final_json']:
            print("\n📄 最终JSON格式已生成完成")
        else:
            print("\n📋 最终计划已生成完成")

        print("=" * 60)

    except Exception as e:
        logging.error(f"代理执行失败: {e}")
        raise


if __name__ == '__main__':

    asyncio.run(main())