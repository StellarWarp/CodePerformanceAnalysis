import statistics
from collections import defaultdict
from typing import Dict, Optional
from typing import List, Any

import anytree
from pydantic import BaseModel, Field
from pydantic import ValidationError
from typing_extensions import Annotated


class CallStackFrame(BaseModel):
    """
    定义了调用栈中单个帧的数据结构。
    """
    event_id: Annotated[
        int,
        Field(description="A unique identifier for the event within the entire trace.")
    ] = -1

    TimerName: Annotated[
        Optional[str],
        Field(description='The name of the timer. For nodes filtered by this function, it is typically "Frame".')
    ] = None

class CallEventMeta(BaseModel):
    """
    CallEventMeta model defined using Pydantic and Annotated.
    Each field includes a detailed description, which becomes part of the model's schema.
    """
    ThreadId: Annotated[
        int,
        Field(description="The ID of the thread that executed this event.")
    ] = -1

    ThreadName: Annotated[
        Optional[str],
        Field(description="The name of the thread that executed this event.")
    ] = None

    TimerId: Annotated[
        int,
        Field(description="The ID of the timer.")
    ] = -1

    TimerName: Annotated[
        Optional[str],
        Field(description='The name of the timer. For nodes filtered by this function, it is typically "Frame".')
    ] = None

    StartTime: Annotated[
        float,
        Field(description="Timestamp when the event started, in seconds.")
    ] = -1.0

    EndTime: Annotated[
        float,
        Field(description="Timestamp when the event ended, in seconds.")
    ] = -1.0

    Duration: Annotated[
        float,
        Field(description="The duration of the event (EndTime - StartTime), in seconds.")
    ] = 0.0

    CallDepth: Annotated[
        int,
        Field(description="The depth of the event in the call stack.")
    ] = 0

    event_id: Annotated[
        int,
        Field(description="A unique identifier for the event within the entire trace.")
    ] = -1

    call_stack: Annotated[
        List[CallStackFrame],
        Field(
            description="A list of call stack frames showing the complete call stack for that node (excluding the event itself) in order from top to bottom (caller -> callee), with each element containing the event id and timer name of a stack")
    ]


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
        # 步骤 B: 调用父类 (anytree.Node) 的初始化方法来构建树。
        super().__init__(name, parent)
        full_path = list(self.path)[1:-1] # TODO 去除虚构的根节点和该节点自身
        call_stack = [CallStackFrame(event_id=node.event_id,TimerName=node.TimerName) for node in full_path]
        # 步骤 A: 将所有额外的关键字参数用于创建 self.meta 实例。
        # Pydantic 会自动验证传入的 kwargs，如果有多余或类型错误的参数，会在这里报错。
        self.meta = CallEventMeta(**kwargs,call_stack=call_stack)



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
        Field(description="A unique identifier for the event within the entire trace.")
    ]

    ThreadName: Annotated[
        Optional[str],
        Field(description="The name of the thread that executed this event.")
    ]

    Duration: Annotated[
        float,
        Field(description="The duration of the event (EndTime - StartTime), in seconds.")
    ]

    StartTime: Annotated[
        float,
        Field(description="Timestamp when the event started, in seconds.")
    ]

    EndTime: Annotated[
        float,
        Field(description="Timestamp when the event ended, in seconds.")
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
              description="The name of the timer, typically a function or a scoped event. (计时器的名称，通常是函数名或作用域事件。)")
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


