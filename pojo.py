import os.path
import statistics
from collections import defaultdict
from typing import Dict, Optional
from typing import List, Any

import anytree
from pydantic import BaseModel, Field
from pydantic import ValidationError
from typing_extensions import Annotated


TIMER_DESCRIPTION = """由Unreal Insights捕获的计时事件（Timer）的名称。这是在Timing Insights视图中看到的性能测量范围的核心标识符。该名称通常在引擎的C++源代码中通过性能分析宏（如 SCOPE_CYCLE_COUNTER, SCOPED_NAMED_EVENT）定义，其格式多样，主要有以下几种情况：

1.  **标准函数签名**: 最常见的格式，直接对应一个C++类和函数名。格式为 `ClassName::FunctionName`。示例: `FScene::Render`。

2.  **带源码位置的名称**: 在某些情况下，为了便于调试，名称会附加其在源代码中的精确位置。格式为 `Name [FileName.cpp(LineNumber)]`。示例: `SSceneOutlinerTreeView [SSceneOutliner.cpp(265)]`。

3.  **描述性/分类名称**: 用于标记一个更广泛的系统或任务，而不仅仅是一个函数。这种名称通常提供了额外的上下文信息。格式可能为 `Category::SubCategory (Description)`。示例: `Slate::Tick (Time and Widgets)`。

该字段是将性能数据与特定代码块或引擎模块关联起来的关键。在进行自动化分析时，需要考虑其格式的多样性来进行解析。"""

class CallStackFrame(BaseModel):
    """
    定义了调用栈中单个帧的数据结构。
    """
    event_id: Annotated[
        int,
        Field(description="整个跟踪中事件的唯一标识符。")
    ] = -1

    TimerName: Annotated[
        str,
        Field(description=TIMER_DESCRIPTION)
    ] = None


class TimerSource(BaseModel):
    source_file: Annotated[
        str,
        Field(
            description="定义此计时器(Timer)的源代码文件的绝对或相对路径。"
        )
    ]

    source_line: Annotated[
        int,
        Field(
            description="在源代码文件中定义此计时器(Timer)的具体行号。"
        )
    ]


class CallEventMeta(BaseModel):
    """
    CallEventMeta model defined using Pydantic and Annotated.
    Each field includes a detailed description, which becomes part of the model's schema.
    """
    ThreadId: Annotated[
        int,
        Field(description="执行此事件的线程ID。")
    ] = -1

    ThreadName: Annotated[
        str,
        Field(description="执行此事件的线程的名称。")
    ] = None

    TimerId: Annotated[
        int,
        Field(description="定时器的ID。")
    ] = -1

    TimerName: Annotated[
        str,
        Field(description=TIMER_DESCRIPTION)
    ] = None

    StartTime: Annotated[
        float,
        Field(description="事件开始的时间戳，单位为秒。")
    ] = -1.0

    EndTime: Annotated[
        float,
        Field(description="事件结束的时间戳，单位为秒。")
    ] = -1.0

    Duration: Annotated[
        float,
        Field(description="事件的持续时间（EndTime - StartTime），单位为秒。")
    ] = 0.0

    CallDepth: Annotated[
        int,
        Field(description="事件在调用栈中的深度。")
    ] = 0

    event_id: Annotated[
        int,
        Field(description="整个跟踪中事件的唯一标识符。")
    ] = -1

    call_stack: Annotated[
        List[CallStackFrame],
        Field(
            description="调用堆栈帧的列表，从上到下（caller -> callee）显示该节点的完整调用堆栈（不包括事件本身），每个元素包含一个堆栈的事件id和定时器名称")
    ]

    timer_source_info: Annotated[
        Optional[TimerSource],
        Field(
            description="表示一个性能计时器 (Timer) 在源代码中的定义位置, 包含源文件的绝对路径或相对路径以及行号。如果为空(null)，则代表无法获取该计时器在源码中的位置。"
        )
    ] = None





class CallEventNode(anytree.Node):
    """
    一个 anytree 节点，它通过组合方式 **拥有**一个 CallEventMeta 实例来存储元数据。
    """
    # 声明 meta 属性的类型，以便类型检查器和 IDE 能识别它
    meta: CallEventMeta

    def __init__(self, name: str, parent: Optional['CallEventNode'] = None, **kwargs: Any):
        """
        初始化方法：
        1. 将所有元数据相关的关键字参数 (kwargs) 传递给 CallEventMeta 进行验证和实例化。
        2. 使用 name 和 parent 初始化 anytree.Node。
        """
        super().__init__(name, parent)
        source_file = kwargs.pop('source_file', None)
        source_line = kwargs.pop('source_line', -1)
        timer_source_info = None
        if isinstance(source_file, str) and source_file.strip() and source_line != -1:
            timer_source_info = TimerSource(
                source_file=source_file,
                source_line=source_line,
            )
        path_for_stack = list(self.path)[1:-1]
        call_stack = [
            CallStackFrame(event_id=node.meta.event_id, TimerName=node.meta.TimerName)
            for node in path_for_stack
        ]
        self.meta = CallEventMeta(
            **kwargs,
            call_stack=call_stack,
            timer_source_info=timer_source_info
        )



    def __getattr__(self, name: str) -> Any:
        # ✅ FIX: Access model_fields from the CLASS, not the instance.
        if name in CallEventMeta.model_fields:
            return getattr(self.meta, name)

        raise AttributeError(f"'{type(self).__name__}' object and its 'meta' attribute have no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        # Check if 'meta' has been initialized and if the attribute belongs to the meta model's schema.
        # ✅ FIX: Access model_fields from the CLASS, not the instance.
        if 'meta' in self.__dict__ and name in CallEventMeta.model_fields:
            try:
                setattr(self.meta, name, value)
            except ValidationError as e:
                raise ValueError(f"Validation error setting '{name}': {e}") from e
        else:
            # Handle standard attributes like 'name', 'parent', 'meta', etc.
            super().__setattr__(name, value)



class ExceptionFrame(BaseModel):
    event_id: Annotated[
        int,
        Field(description="整个跟踪中事件的唯一标识符。")
    ]

    ThreadName: Annotated[
        str,
        Field(description="执行此事件的线程的名称。")
    ]

    Duration: Annotated[
        float,
        Field(description="事件的持续时间（EndTime - StartTime），单位为秒。")
    ]

    StartTime: Annotated[
        float,
        Field(description="事件开始的时间戳，单位为秒。")
    ]

    EndTime: Annotated[
        float,
        Field(description="事件结束的时间戳，单位为秒。")
    ]


class TimerMeta(BaseModel):
    """
    Represents the metadata for a single timer from an Unreal Insights 'TimerStat.csv' report.
    This model is designed to be constructed directly from a dictionary representing a row in the CSV.
    All time values are in seconds (s).

    代表来自 Unreal Insights 'TimerStat.csv' 报告中单个计时器的元数据。
    该模型设计用于直接从代表CSV中一行的字典进行构造。
    所有时间单位均为秒 (s)。
    """
    TimerName: Annotated[
        str,
        Field(
              description=TIMER_DESCRIPTION)
    ]

    count: Annotated[
        int,
        Field(
              description="The number of times this timer was hit during the trace. (在跟踪期间该计时器被触发的次数。)")
    ]

    inclusive_total: Annotated[
        float,
        Field(
              description="Total inclusive time in seconds (s). This is the sum of time spent in this timer including all child timers it called. (总包含时间，单位为秒 (s)。这是在此计时器及其调用的所有子计时器中所花费时间的总和。)")
    ]

    inclusive_min: Annotated[
        float,
        Field(
              description="Minimum inclusive time for a single instance of this timer, in seconds (s). (单次调用的最小包含时间，单位为秒 (s)。)")
    ]

    inclusive_max: Annotated[
        float,
        Field(
              description="Maximum inclusive time for a single instance of this timer, in seconds (s). (单次调用的最大包含时间，单位为秒 (s)。)")
    ]

    inclusive_avg: Annotated[
        float,
        Field(
              description="Average inclusive time for this timer, in seconds (s). (该计时器的平均包含时间，单位为秒 (s)。)")
    ]

    inclusive_median: Annotated[
        float,
        Field(
              description="Median inclusive time for this timer, in seconds (s). (该计时器的中位数包含时间，单位为秒 (s)。)")
    ]

    exclusive_total: Annotated[
        float,
        Field(description="Total exclusive time in seconds (s). This is the sum of time spent in this timer, excluding time spent in any child timers it called. (总独占时间，单位为秒 (s)。这是在此计时器自身花费的时间总和，不包括在其调用的任何子计时器中花费的时间。)")
    ]

    exclusive_min: Annotated[
        float,
        Field(description="Minimum exclusive time for a single instance of this timer, in seconds (s). (单次调用的最小独占时间，单位为秒 (s)。)")
    ]

    exclusive_max: Annotated[
        float,
        Field(
              description="Maximum exclusive time for a single instance of this timer, in seconds (s). (单次调用的最大独占时间，单位为秒 (s)。)")
    ]

    exclusive_avg: Annotated[
        float,
        Field(
              description="Average exclusive time for this timer, in seconds (s). (该计时器的平均独占时间，单位为秒 (s)。)")
    ]

    exclusive_median: Annotated[
        float,
        Field(
              description="Median exclusive time for this timer, in seconds (s). (该计时器的中位数独占时间，单位为秒 (s)。)")
    ]





class CostDistribution(BaseModel):
    """
    用于分析和统计调用栈中各个Timer的性能开销分布。
    基于Unreal Insights的性能分析概念，提供详细的性能统计信息。
    """


    cost_distribution: Annotated[
        Dict[str, TimerMeta],
        Field(description="以TimerName为键的性能开销分布字典，包含每个Timer的详细统计信息")
    ]

    total_analysis_time: Annotated[
        float,
        Field(description="整个分析时间窗口的总时长（秒）")
    ] = 0.0

    total_events_count: Annotated[
        int,
        Field(description="分析时间窗口内的总事件数量")
    ] = 0

    @staticmethod
    def calculate_cost_distribution(node: 'CallEventNode', max_depth: Optional[int] = None) -> 'CostDistribution':
        """
        计算给定节点及其子树的性能开销分布。

        Args:
            node: 要分析的CallEventNode节点，将分析该节点及其所有子节点
            max_depth: 最大分析深度，None表示无深度限制。深度从给定节点开始计算，
                      例如max_depth=1只分析当前节点，max_depth=2分析当前节点和直接子节点

        Returns:
            CostDistribution: 包含性能分析结果的对象
        """
        # 用于存储每个Timer的原始数据
        timer_data = defaultdict(lambda: {
            'inclusive_times': [],
            'exclusive_times': [],
            'count': 0
        })

        # 计算分析时间窗口的总时长和总事件计数
        total_analysis_time = 0.0
        total_events_count = 0

        # 根据深度限制确定要分析的节点
        if max_depth is not None:
            nodes_to_process = [n for n in anytree.PreOrderIter(node, maxlevel=max_depth)]
        else:
            nodes_to_process = list(anytree.PreOrderIter(node))

        # 遍历节点树，收集所有Timer的性能数据
        for current_node in nodes_to_process:
            if hasattr(current_node, 'meta') and current_node.meta.TimerName:
                timer_name = current_node.meta.TimerName
                inclusive_duration = current_node.meta.Duration

                # 计算exclusive时间（不包括子节点的时间）
                if max_depth is not None:
                    current_depth = current_node.depth - node.depth
                    # 如果还在深度限制内，考虑子节点；否则将所有时间视为exclusive时间
                    if current_depth < max_depth - 1:
                        children_duration = sum(
                            child.meta.Duration
                            for child in current_node.children
                            if hasattr(child, 'meta') and child.meta.Duration > 0
                        )
                    else:
                        children_duration = 0.0  # 已达到最大深度，所有时间都是exclusive的
                else:
                    children_duration = sum(
                        child.meta.Duration
                        for child in current_node.children
                        if hasattr(child, 'meta') and child.meta.Duration > 0
                    )

                exclusive_duration = max(0.0, inclusive_duration - children_duration)

                # 记录数据
                timer_data[timer_name]['inclusive_times'].append(inclusive_duration)
                timer_data[timer_name]['exclusive_times'].append(exclusive_duration)
                timer_data[timer_name]['count'] += 1

                total_events_count += 1

        # 计算总分析时间
        if hasattr(node, 'meta') and node.meta.Duration > 0:
            total_analysis_time = node.meta.Duration
        else:
            # 使用时间范围计算
            max_end_time = 0.0
            min_start_time = float('inf')

            for current_node in nodes_to_process:
                if hasattr(current_node, 'meta'):
                    if current_node.meta.StartTime >= 0:
                        min_start_time = min(min_start_time, current_node.meta.StartTime)
                    if current_node.meta.EndTime >= 0:
                        max_end_time = max(max_end_time, current_node.meta.EndTime)

            if min_start_time != float('inf') and max_end_time > min_start_time:
                total_analysis_time = max_end_time - min_start_time
            else:
                # 回退到所有inclusive时间的总和
                total_analysis_time = sum(
                    sum(data['inclusive_times']) for data in timer_data.values()
                )

        # 构建TimerMeta对象
        cost_distribution = {}

        for timer_name, data in timer_data.items():
            inclusive_times = data['inclusive_times']
            exclusive_times = data['exclusive_times']
            count = data['count']

            if count == 0:
                continue

            # 计算inclusive统计信息
            inclusive_total = sum(inclusive_times)
            inclusive_min = min(inclusive_times)
            inclusive_max = max(inclusive_times)
            inclusive_avg = inclusive_total / count
            inclusive_median = statistics.median(inclusive_times)

            # 计算exclusive统计信息
            exclusive_total = sum(exclusive_times)
            exclusive_min = min(exclusive_times) if exclusive_times else 0.0
            exclusive_max = max(exclusive_times) if exclusive_times else 0.0
            exclusive_avg = exclusive_total / count
            exclusive_median = statistics.median(exclusive_times) if exclusive_times else 0.0

            # 创建TimerMeta对象
            timer_meta = TimerMeta(
                TimerName=timer_name,
                count=count,
                inclusive_total=inclusive_total,
                inclusive_min=inclusive_min,
                inclusive_max=inclusive_max,
                inclusive_avg=inclusive_avg,
                inclusive_median=inclusive_median,
                exclusive_total=exclusive_total,
                exclusive_min=exclusive_min,
                exclusive_max=exclusive_max,
                exclusive_avg=exclusive_avg,
                exclusive_median=exclusive_median
            )

            cost_distribution[timer_name] = timer_meta

        return CostDistribution(
            cost_distribution=cost_distribution,
            total_analysis_time=total_analysis_time,
            total_events_count=total_events_count
        )

    def get_top_timers_by_inclusive_total(self, top_n: int = 10) -> List[TimerMeta]:
        """
        获取按总包含时间排序的前N个Timer。

        Args:
            top_n: 返回的Timer数量

        Returns:
            按总包含时间降序排列的TimerMeta列表
        """
        return sorted(
            self.cost_distribution.values(),
            key=lambda timer: timer.inclusive_total,
            reverse=True
        )[:top_n]

    def get_top_timers_by_exclusive_total(self, top_n: int = 10) -> List[TimerMeta]:
        """
        获取按总独占时间排序的前N个Timer。

        Args:
            top_n: 返回的Timer数量

        Returns:
            按总独占时间降序排列的TimerMeta列表
        """
        return sorted(
            self.cost_distribution.values(),
            key=lambda timer: timer.exclusive_total,
            reverse=True
        )[:top_n]

    def get_top_timers_by_count(self, top_n: int = 10) -> List[TimerMeta]:
        """
        获取按调用次数排序的前N个Timer。

        Args:
            top_n: 返回的Timer数量

        Returns:
            按调用次数降序排列的TimerMeta列表
        """
        return sorted(
            self.cost_distribution.values(),
            key=lambda timer: timer.count,
            reverse=True
        )[:top_n]

    def get_top_timers_by_inclusive_avg(self, top_n: int = 10) -> List[TimerMeta]:
        """
        获取按平均包含时间排序的前N个Timer。

        Args:
            top_n: 返回的Timer数量

        Returns:
            按平均包含时间降序排列的TimerMeta列表
        """
        return sorted(
            self.cost_distribution.values(),
            key=lambda timer: timer.inclusive_avg,
            reverse=True
        )[:top_n]

    def get_timer_stats(self, timer_name: str) -> Optional[TimerMeta]:
        """
        获取指定Timer的统计信息。

        Args:
            timer_name: Timer名称

        Returns:
            对应的TimerMeta对象，如果不存在则返回None
        """
        return self.cost_distribution.get(timer_name)

    def get_total_inclusive_ratio(self, timer_name: str) -> float:
        """
        获取指定Timer的包含时间占总分析时间的比例。

        Args:
            timer_name: Timer名称

        Returns:
            时间占比（0.0-1.0），如果Timer不存在或总时间为0则返回0.0
        """
        timer = self.get_timer_stats(timer_name)
        if timer is None or self.total_analysis_time <= 0:
            return 0.0
        return timer.inclusive_total / self.total_analysis_time

    def get_total_exclusive_ratio(self, timer_name: str) -> float:
        """
        获取指定Timer的独占时间占总分析时间的比例。

        Args:
            timer_name: Timer名称

        Returns:
            时间占比（0.0-1.0），如果Timer不存在或总时间为0则返回0.0
        """
        timer = self.get_timer_stats(timer_name)
        if timer is None or self.total_analysis_time <= 0:
            return 0.0
        return timer.exclusive_total / self.total_analysis_time

    def print_summary(self, top_n: int = 10):
        """
        打印性能分析摘要信息。

        Args:
            top_n: 显示的热点Timer数量
        """
        print(f"=== 性能分析摘要 ===")
        print(f"总分析时间: {self.total_analysis_time:.6f}秒")
        print(f"总事件数量: {self.total_events_count}")
        print(f"唯一Timer数量: {len(self.cost_distribution)}")
        print()

        print(f"=== 按总包含时间排序的前{top_n}个Timer ===")
        top_timers = self.get_top_timers_by_inclusive_total(top_n)
        for i, timer in enumerate(top_timers, 1):
            ratio = self.get_total_inclusive_ratio(timer.TimerName)
            print(f"{i:2d}. {timer.TimerName:<40} "
                  f"总时间: {timer.inclusive_total:10.6f}s "
                  f"占比: {ratio * 100:6.2f}% "
                  f"次数: {timer.count:6d} "
                  f"平均: {timer.inclusive_avg:10.6f}s")

        print()
        print(f"=== 按总独占时间排序的前{top_n}个Timer ===")
        top_exclusive_timers = self.get_top_timers_by_exclusive_total(top_n)
        for i, timer in enumerate(top_exclusive_timers, 1):
            ratio = self.get_total_exclusive_ratio(timer.TimerName)
            print(f"{i:2d}. {timer.TimerName:<40} "
                  f"独占时间: {timer.exclusive_total:10.6f}s "
                  f"占比: {ratio * 100:6.2f}% "
                  f"次数: {timer.count:6d} "
                  f"平均: {timer.exclusive_avg:10.6f}s")












class WorkFlowNode(BaseModel):
    task_description: Annotated[str, "工作流节点任务描述"]
    task_name: Annotated[str, "工作流节点名称,每个工作流节点的唯一标识符"]
    dependencies: Annotated[List[str],'前置依赖的工作流节点名称']
    input_specification: Annotated[str, '输入']
    output_specification: Annotated[str, '输出']
    required_tools: Annotated[List[str],'所需工具列表名称']



class TaskPrompt:
    """
    用于动态构建和组织“任务执行Agent”的System Prompt。
    它将一个复杂的指令分解为多个结构化部分，在任务开始前渲染成最终的System Prompt，
    为执行Agent提供其角色、目标、工具、上下文等所有必要信息。
    """
    # --- 用于渲染 System Prompt 的核心组件 ---
    project_info_prompt: Annotated[Optional[str], Field(
        description='''项目元信息：定义整个工作流的宏观背景、核心业务或最终目标，为执行Agent提供顶层上下文。'''
    )] = None

    experience_prompt: Annotated[Optional[str], Field(
        description='''相关经验或知识库：注入与当前任务相关的背景知识、最佳实践或历史解决方案，作为Agent决策的参考。'''
    )] = None

    important_prompt: Annotated[Optional[str], Field(
        description='''重要指令或约束：需要特别强调的关键指令、规则或必须遵守的约束条件。'''
    )] = None

    helper_assistant_prompt: Annotated[Optional[str], Field(
        description='''协作单元介绍：说明与当前Agent协同工作的其他AI Agent或外部系统（如有），并描述它们的作用。'''
    )] = None

    # node_struct_prompt: Annotated[Optional[str], Field(
    #     description='''工作流节点结构定义：描述工作流中一个标准节点的通用数据结构，帮助Agent理解其自身及相邻节点的构成。'''
    # )] = None
    #
    # role_prompt: Annotated[Optional[str], Field(
    #     description='''角色定义提示：明确告知任务执行Agent它当前应当扮演的具体角色（例如：“你是一个专注代码分析的AI助手”）。'''
    # )] = None
    #
    # specific_task_prompt: Annotated[Optional[str], Field(
    #     description='''本节点核心任务描述：清晰、详尽地描述当前节点需要完成的核心目标（Node Objective），这是Agent的首要任务。'''
    # )] = None
    #
    # tool_description_prompt: Annotated[Optional[str], Field(
    #     description='''可用工具集描述：提供给当前任务执行Agent的所有可用工具的列表及其详细描述，严格限定其行动范围。'''
    # )] = None
    #
    #
    #
    # response_prompt: Annotated[Optional[str], Field(
    #     description='''输出格式与契约：严格限定Agent完成任务后需要输出的内容格式（如：必须符合特定的JSON Schema），定义其“数据契约”。'''
    # )] = None
    #
    # extra_prompt: Annotated[Optional[str], Field(
    #     description='''其他补充信息：用于提供未在其他字段中涵盖的任何额外上下文、边缘案例说明或补充指令。'''
    # )] = None
    #
    # partial_nodes_prompt: Annotated[Optional[str], Field(
    #     description='''相邻节点上下文：提供当前节点的直接上游（父节点）和下游（子节点）的摘要信息，帮助Agent理解其在工作流中的位置和数据流关系。'''
    # )] = None
    #
    # # --- 用于生成 User Message 的核心组件 ---
    # task_message_prompt: Annotated[Optional[str], Field(
    #     description='''任务启动指令：作为User Message发送给Agent的启动信号，通常包含对任务的最终、最直接的指令，例如“开始执行任务：Identify_Macro_Bottleneck”。'''
    # )] = None


class SubTaskItem(BaseModel):
    """
    代表为完成一个工作流节点而拆分出的一个更细粒度的、可执行的子任务。
    这是任务执行Agent内部进行任务分解后的基本工作单元。
    """
    task: Annotated[str, Field(
        description='''子任务指令：一个清晰、原子化的指令，明确定义了该子任务需要达成的具体目标和预期的产出形式。'''
    )]

    context: Annotated[str, Field(
        description='''子任务上下文：执行此子任务所需的所有背景信息，包括但不限于所需的数据片段、来自前序子任务的输出、执行此任务的原因等。'''
    )]


class Task(BaseModel):
    """
    一个工作流节点的完整“任务包（Task Package）”。
    它封装了将一个规划好的工作流节点交付给“任务执行Agent”所需的所有信息，
    是连接“规划层”和“执行层”的核心数据对象。
    """
    task_prompt: Annotated[TaskPrompt, Field(
        description='''任务执行Agent的Prompt配置：包含了用于生成该Agent System Prompt的所有结构化信息。'''
    )]

    # 注意：这里的sub_task_list可以由规划Agent预先进行粗粒度拆分，也可以为空，由执行Agent在运行时自行拆分。
    sub_task_list: Annotated[List[SubTaskItem], Field(
        description='''预设的顶级子任务列表：为完成本节点的核心目标而预先规划的一系列高阶步骤。执行Agent可以遵循此列表，也可以在此基础上进一步分解。'''
    )]

    mcp_tools: Annotated[Dict[str,List[str]], Field(
        description='''执行该工作流节点所需要的MCP工具列表，key代表mcp_server_id,value为该MCP Server下所需要的工具列表名称'''
    )]



class WorkFlowContextManager:

    context: Dict[str, str]


class TaskGenerate(Task):
    # 该任务将创建一个任务生成Agent，该Agent的功能是每个工作流节点创建一个Task类，该Agent首先接受任务规划Agent输出的工作流，将每个工作流节点转换成Task类是它的顶级子任务（一个工作流节点一个顶级子任务），
    # 每个子任务都需要检索相关优化知识（到时候执行该工作流节点的Agent需要的优化知识）并为到时候需要执行该工作流节点的Agent指定初始的顶级子任务。
    # 任务生成Agent的每个顶级子任务的输出是一个Task类，最终该Agent应当返回一个Task列表

    @staticmethod
    def generate_task1(workFlowNode:WorkFlowNode)->Task:
        # 搜索相关经验文档并返回字符串
        # 制定顶级子任务
        pass

















