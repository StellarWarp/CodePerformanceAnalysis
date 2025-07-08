from typing import Annotated, List

from anytree import findall, find_by_attr
from fastmcp import FastMCP
from pydantic import Field

import dataloader
import opt_search
from pojo import CallEventMeta, ExceptionFrame, CostDistribution

mcp = FastMCP(name="Call Tree MCP Server")
SESSION_BUFFER = dataloader.TraceDataManager(r'C:\Program Files\Epic Games\UE_5.6\Engine\Binaries\Win64\UnrealInsights.exe')

@mcp.tool(
    name="get_exception_frame",
    description="""Identifies and retrieves exception frames from Unreal Insights trace data. Exception frames are defined as Frame events that exceed a specified duration threshold, typically indicating performance bottlenecks or unusually long frame times that may cause framerate drops or stuttering in the application.

从Unreal Insights跟踪数据中识别并检索异常帧。异常帧定义为超过指定持续时间阈值的Frame事件，通常表示性能瓶颈或异常长的帧时间，可能导致应用程序的帧率下降或卡顿。""",
    tags={'performance', 'frame_analysis', 'exception_detection'},
)
def get_exception_frame(
        utrace_file: Annotated[str, Field(description="The path or unique identifier of the utrace file to analyze. | 要分析的utrace文件的路径或唯一标识符。")],
        max_frame_cost: Annotated[float, Field(description="The minimum duration threshold (in seconds) to define an exception frame. Frames with duration >= this value will be considered exceptions. | 定义异常帧的最小持续时间阈值（以秒为单位）。持续时间>=此值的帧将被视为异常。")]
) -> Annotated[
    List[ExceptionFrame],
    Field(description="""A list of ExceptionFrame objects representing frames that exceeded the specified duration threshold. Each frame contains event_id, ThreadName, Duration, StartTime, and EndTime information for further analysis and optimization. | 表示超过指定持续时间阈值的帧的ExceptionFrame对象列表。每个帧包含event_id、ThreadName、Duration、StartTime和EndTime信息，用于进一步分析和优化。""")
]:
    root = SESSION_BUFFER[utrace_file][2]
    exception_frames = findall(root,filter_=lambda x: x.name == 'Frame' and x.Duration >= max_frame_cost)

    fields_to_include = {"event_id", "ThreadName", "Duration",'StartTime','EndTime','Duration'}
    result = [frame.meta.model_dump(include=fields_to_include) for frame in exception_frames]
    return result

@mcp.tool(
    name="getKeyNodes",
    description="""Automatically identifies and retrieves key performance-critical events within a specified subtree of the call stack. This tool uses intelligent filtering to discover nodes that contribute significantly to the overall execution time, helping developers focus on the most impactful optimization opportunities. The analysis is based on time cost ratios relative to the subtree root, making it effective for hierarchical performance analysis.

自动识别并检索调用栈指定子树中的关键性能关键事件。此工具使用智能过滤来发现对整体执行时间有重大贡献的节点，帮助开发者专注于最具影响力的优化机会。分析基于相对于子树根的时间成本比率，使其对分层性能分析非常有效。""",
    tags={"performance", "optimization", "call_tree", "hotspot_detection"},
)
def getKeyNodes(
        utrace_file: Annotated[str, Field(description="The path or unique identifier of the utrace file to analyze. | 要分析的utrace文件的路径或唯一标识符。")],
        event_id: Annotated[int, Field(description="The root node event ID of the subtree to query. This serves as the starting point for the key node discovery analysis. | 要查询的子树的根节点事件ID。这作为关键节点发现分析的起始点。")],
        max_threshold: Annotated[float, Field(
            description="The time cost ratio threshold for discovering key nodes (compared to the subtree root node's time cost), default is 0.005 (0.5%). Higher threshold values result in looser filtering conditions, returning more key nodes but potentially including less critical ones. | 用于发现关键节点的时间成本比率阈值（相对于子树根节点的时间成本），默认为0.005（0.5%）。更高的阈值值导致更宽松的过滤条件，返回更多关键节点但可能包括不太关键的节点。")] = 0.005
) -> Annotated[
    List[CallEventMeta],
    Field(description="""A list of CallEventMeta objects representing the key nodes discovered in the subtree. Each object contains comprehensive metadata including timing information, call depth, thread details, and complete call stack information. These nodes represent the most performance-critical events that should be prioritized for optimization. | 表示在子树中发现的关键节点的CallEventMeta对象列表。每个对象包含全面的元数据，包括时间信息、调用深度、线程详细信息和完整的调用栈信息。这些节点代表应优先进行优化的最关键性能事件。""")
]:
    root = SESSION_BUFFER[utrace_file][2]
    node = find_by_attr(root, event_id, name='event_id')
    opd = opt_search.OptimizedCandidateDiscovery(node, max_threshold)
    key_nodes = opd.getKeyNodes()

    return [n.meta for n in key_nodes]

@mcp.tool(
    name="getNodeMetaInfo",
    description="""Retrieves comprehensive metadata information for a specific event node identified by its event ID. This includes detailed timing information, thread context, call stack hierarchy, and execution details. Essential for deep-dive analysis of specific performance events and understanding their execution context within the broader call tree.

检索由其事件ID标识的特定事件节点的综合元数据信息。这包括详细的时间信息、线程上下文、调用栈层次结构和执行详细信息。对于特定性能事件的深入分析以及理解它们在更广泛调用树中的执行上下文至关重要。""",
    tags={"metadata", "event_analysis", "call_stack"},
)
def getNodeMetaInfo(
        utrace_file: Annotated[str, Field(description="The path or unique identifier of the utrace file to analyze. | 要分析的utrace文件的路径或唯一标识符。")],
        event_id: Annotated[int, Field(description="The unique ID of the event node to query. This ID uniquely identifies a specific event within the trace data. | 要查询的事件节点的唯一ID。此ID在跟踪数据中唯一标识特定事件。")]
) -> Annotated[
    CallEventMeta,
    Field(description="""A CallEventMeta object containing comprehensive metadata for the specified event node. This includes timing information (StartTime, EndTime, Duration), thread details (ThreadId, ThreadName), timer information (TimerId, TimerName), call depth, and the complete call stack hierarchy leading to this event. | 包含指定事件节点综合元数据的CallEventMeta对象。这包括时间信息（StartTime、EndTime、Duration）、线程详细信息（ThreadId、ThreadName）、计时器信息（TimerId、TimerName）、调用深度以及导致此事件的完整调用栈层次结构。""")
]:
    root = SESSION_BUFFER[utrace_file][2]
    node = find_by_attr(root, event_id, name='event_id')
    return node.meta


@mcp.tool(
    name="getCostDistribution",
    description="""Calculates and retrieves comprehensive performance cost distribution analysis for a specified subtree. This analysis provides detailed statistical information about all timers within the subtree, including inclusive/exclusive timing statistics, call frequencies, and performance ratios. Based on Unreal Insights profiling concepts, it offers insights similar to TimerStat.csv reports but focused on specific call tree branches.

计算并检索指定子树的综合性能成本分布分析。此分析提供子树内所有计时器的详细统计信息，包括包含/独占时间统计、调用频率和性能比率。基于Unreal Insights分析概念，它提供类似于TimerStat.csv报告的见解，但专注于特定的调用树分支。""",
    tags={"performance", "statistics", "cost_analysis", "timer_distribution"},
)
def getCostDistribution(
        utrace_file: Annotated[str, Field(description="The path or unique identifier of the utrace file to analyze. | 要分析的utrace文件的路径或唯一标识符。")],
        event_id: Annotated[int, Field(description="The unique ID of the event node to use as the root for cost distribution analysis. The analysis will include this node and all its descendants in the call tree. | 用作成本分布分析根节点的事件节点的唯一ID。分析将包括此节点及其在调用树中的所有后代。")]
) -> Annotated[
    CostDistribution,
    Field(description="""A CostDistribution object containing comprehensive performance analysis results for the specified subtree. This includes a dictionary of TimerMeta objects (keyed by timer name) with detailed statistics for each timer including count, inclusive/exclusive timing data (total, min, max, average, median), total analysis time, and total event count. The data can be used to identify performance hotspots, understand timing distributions, and prioritize optimization efforts. | 包含指定子树综合性能分析结果的CostDistribution对象。这包括TimerMeta对象字典（按计时器名称键控），每个计时器的详细统计信息包括计数、包含/独占时间数据（总计、最小、最大、平均、中位数）、总分析时间和总事件计数。数据可用于识别性能热点、理解时间分布并优先考虑优化工作。""")
]:
    root = SESSION_BUFFER[utrace_file][2]
    node = find_by_attr(root, event_id, name='event_id')
    return CostDistribution.calculate_cost_distribution(node)




if __name__ == "__main__":
    mcp.run(transport="sse",port=8001)