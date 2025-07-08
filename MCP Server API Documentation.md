Here is the API documentation for the MCP server, provided in both English and Chinese.



# MCP Server API Documentation





## Introduction



This document provides the API documentation for the MCP (Multi-purpose Cooperative Processing) server. This server is designed to analyze performance data from Unreal Engine's `.utrace` files. By leveraging Unreal Insights for automated trace analysis, the server offers a suite of tools to identify performance bottlenecks, analyze frame-by-frame data, and generate detailed cost distribution statistics.

This API is built using Python with the `fastmcp` library for creating tools, `pydantic` for data modeling and validation, and `anytree` for representing call stack data hierarchically. It is intended for developers and performance engineers who need to programmatically analyze Unreal Engine performance captures.

------



## Data Models (POCOs - Plain Old C# Objects)



These are the primary data structures used throughout the API.



### `CallStackFrame`



Defines the data structure for a single frame in the call stack.

| Field       | Type              | Description                                                  |
| ----------- | ----------------- | ------------------------------------------------------------ |
| `event_id`  | integer           | A unique identifier for the event within the entire trace.   |
| `TimerName` | string (optional) | The name of the timer. For nodes filtered by this function, it is typically "Frame". |



### `CallEventMeta`



Contains the metadata for a call event, providing detailed information about its execution.

| Field        | Type                   | Description                                                  |
| ------------ | ---------------------- | ------------------------------------------------------------ |
| `ThreadId`   | integer                | The ID of the thread that executed this event.               |
| `ThreadName` | string (optional)      | The name of the thread that executed this event.             |
| `TimerId`    | integer                | The ID of the timer.                                         |
| `TimerName`  | string (optional)      | The name of the timer. For nodes filtered by this function, it is typically "Frame". |
| `StartTime`  | float                  | Timestamp when the event started, in seconds.                |
| `EndTime`    | float                  | Timestamp when the event ended, in seconds.                  |
| `Duration`   | float                  | The duration of the event (EndTime - StartTime), in seconds. |
| `CallDepth`  | integer                | The depth of the event in the call stack.                    |
| `event_id`   | integer                | A unique identifier for the event within the entire trace.   |
| `call_stack` | List[`CallStackFrame`] | A list of call stack frames showing the complete call stack for that node (excluding the event itself) in order from top to bottom (caller -> callee), with each element containing the event id and timer name of a stack. |



### `CallEventNode`



An `anytree` node that holds a `CallEventMeta` instance to store metadata. It provides a hierarchical representation of call events.



### `ExceptionFrame`



Represents a frame that has exceeded a defined performance threshold, indicating a potential performance issue.

| Field        | Type              | Description                                                  |
| ------------ | ----------------- | ------------------------------------------------------------ |
| `event_id`   | integer           | A unique identifier for the event within the entire trace.   |
| `ThreadName` | string (optional) | The name of the thread that executed this event.             |
| `Duration`   | float             | The duration of the event (EndTime - StartTime), in seconds. |
| `StartTime`  | float             | Timestamp when the event started, in seconds.                |
| `EndTime`    | float             | Timestamp when the event ended, in seconds.                  |



### `TimerMeta`



Represents the metadata for a single timer, analogous to a row in Unreal Insights' `TimerStat.csv` report. All time values are in seconds.

| Field              | Type    | Description                                                  |
| ------------------ | ------- | ------------------------------------------------------------ |
| `TimerName`        | string  | The name of the timer, typically a function or a scoped event. |
| `count`            | integer | The number of times this timer was hit during the trace.     |
| `inclusive_total`  | float   | Total inclusive time in seconds (s). This is the sum of time spent in this timer including all child timers it called. |
| `inclusive_min`    | float   | Minimum inclusive time for a single instance of this timer, in seconds (s). |
| `inclusive_max`    | float   | Maximum inclusive time for a single instance of this timer, in seconds (s). |
| `inclusive_avg`    | float   | Average inclusive time for this timer, in seconds (s).       |
| `inclusive_median` | float   | Median inclusive time for this timer, in seconds (s).        |
| `exclusive_total`  | float   | Total exclusive time in seconds (s). This is the sum of time spent in this timer, excluding time spent in any child timers it called. |
| `exclusive_min`    | float   | Minimum exclusive time for a single instance of this timer, in seconds (s). |
| `exclusive_max`    | float   | Maximum exclusive time for a single instance of this timer, in seconds (s). |
| `exclusive_avg`    | float   | Average exclusive time for this timer, in seconds (s).       |
| `exclusive_median` | float   | Median exclusive time for this timer, in seconds (s).        |



### `CostDistribution`



A container for the analysis of performance cost distribution within a call stack, providing detailed performance statistics.

| Field                 | Type                   | Description                                                  |
| --------------------- | ---------------------- | ------------------------------------------------------------ |
| `cost_distribution`   | Dict[str, `TimerMeta`] | A dictionary of performance cost distribution, keyed by TimerName, containing detailed statistics for each timer. |
| `total_analysis_time` | float                  | The total duration of the entire analysis window in seconds. |
| `total_events_count`  | integer                | The total number of events within the analysis window.       |

------



## API Endpoints (Tools)



These are the functions exposed by the MCP server.



### `get_exception_frame`



Identifies and retrieves exception frames from Unreal Insights trace data. Exception frames are defined as "Frame" events that exceed a specified duration threshold, typically indicating performance bottlenecks or unusually long frame times that may cause framerate drops or stuttering in the application.

- **Tags**: `performance`, `frame_analysis`, `exception_detection`

**Parameters:**

| Parameter        | Type   | Description                                                  |
| ---------------- | ------ | ------------------------------------------------------------ |
| `utrace_file`    | string | The path or unique identifier of the `.utrace` file to analyze. |
| `max_frame_cost` | float  | The minimum duration threshold (in seconds) to define an exception frame. Frames with a duration >= this value will be considered exceptions. |

**Returns:**

- **Type**: List[`ExceptionFrame`]
- **Description**: A list of `ExceptionFrame` objects representing frames that exceeded the specified duration threshold. Each frame contains `event_id`, `ThreadName`, `Duration`, `StartTime`, and `EndTime` information for further analysis and optimization.



### `getKeyNodes`



Automatically identifies and retrieves key performance-critical events within a specified subtree of the call stack. This tool uses intelligent filtering to discover nodes that contribute significantly to the overall execution time, helping developers focus on the most impactful optimization opportunities. The analysis is based on time cost ratios relative to the subtree root, making it effective for hierarchical performance analysis.

- **Tags**: `performance`, `optimization`, `call_tree`, `hotspot_detection`

**Parameters:**

| Parameter       | Type    | Default | Description                                                  |
| --------------- | ------- | ------- | ------------------------------------------------------------ |
| `utrace_file`   | string  |         | The path or unique identifier of the `.utrace` file to analyze. |
| `event_id`      | integer |         | The root node event ID of the subtree to query. This serves as the starting point for the key node discovery analysis. |
| `max_threshold` | float   | 0.005   | The time cost ratio threshold for discovering key nodes (compared to the subtree root node's time cost). Higher threshold values result in looser filtering conditions, returning more key nodes but potentially including less critical ones. |

**Returns:**

- **Type**: List[`CallEventMeta`]
- **Description**: A list of `CallEventMeta` objects representing the key nodes discovered in the subtree. Each object contains comprehensive metadata including timing information, call depth, thread details, and complete call stack information. These nodes represent the most performance-critical events that should be prioritized for optimization.



### `getNodeMetaInfo`



Retrieves comprehensive metadata information for a specific event node identified by its `event_id`. This includes detailed timing information, thread context, call stack hierarchy, and execution details. Essential for deep-dive analysis of specific performance events and understanding their execution context within the broader call tree.

- **Tags**: `metadata`, `event_analysis`, `call_stack`

**Parameters:**

| Parameter     | Type    | Description                                                  |
| ------------- | ------- | ------------------------------------------------------------ |
| `utrace_file` | string  | The path or unique identifier of the `.utrace` file to analyze. |
| `event_id`    | integer | The unique ID of the event node to query. This ID uniquely identifies a specific event within the trace data. |

**Returns:**

- **Type**: `CallEventMeta`
- **Description**: A `CallEventMeta` object containing comprehensive metadata for the specified event node. This includes timing information (`StartTime`, `EndTime`, `Duration`), thread details (`ThreadId`, `ThreadName`), timer information (`TimerId`, `TimerName`), call depth, and the complete call stack hierarchy leading to this event.



### `getCostDistribution`



Calculates and retrieves a comprehensive performance cost distribution analysis for a specified subtree. This analysis provides detailed statistical information about all timers within the subtree, including inclusive/exclusive timing statistics, call frequencies, and performance ratios. Based on Unreal Insights profiling concepts, it offers insights similar to `TimerStat.csv` reports but focused on specific call tree branches.

- **Tags**: `performance`, `statistics`, `cost_analysis`, `timer_distribution`

**Parameters:**

| Parameter     | Type    | Description                                                  |
| ------------- | ------- | ------------------------------------------------------------ |
| `utrace_file` | string  | The path or unique identifier of the `.utrace` file to analyze. |
| `event_id`    | integer | The unique ID of the event node to use as the root for the cost distribution analysis. The analysis will include this node and all its descendants in the call tree. |

**Returns:**

- **Type**: `CostDistribution`
- **Description**: A `CostDistribution` object containing comprehensive performance analysis results for the specified subtree. This includes a dictionary of `TimerMeta` objects (keyed by timer name) with detailed statistics for each timer including count, inclusive/exclusive timing data (total, min, max, average, median), total analysis time, and total event count. The data can be used to identify performance hotspots, understand timing distributions, and prioritize optimization efforts.



------



# MCP 服务器 API 文档





## 简介



本文档为 MCP (Multi-purpose Cooperative Processing) 服务器提供了 API 文档。该服务器旨在分析来自虚幻引擎的 `.utrace` 文件的性能数据。通过利用 Unreal Insights 进行自动化追踪分析，该服务器提供了一套工具来识别性能瓶颈、分析逐帧数据并生成详细的成本分布统计信息。

此 API 使用 Python 构建，借助 `fastmcp` 库创建工具，`pydantic` 用于数据建模和验证，`anytree` 用于分层表示调用堆栈数据。它适用于需要以编程方式分析虚幻引擎性能捕获的开发人员和性能工程师。

------



## 数据模型 (POCOs)



这些是整个 API 中使用的主要数据结构。



### `CallStackFrame`



定义了调用栈中单个帧的数据结构。

| 字段        | 类型              | 描述                                                 |
| ----------- | ----------------- | ---------------------------------------------------- |
| `event_id`  | integer           | 事件在整个追踪文件中的唯一标识符。                   |
| `TimerName` | string (optional) | 计时器的名称。对于此函数筛选的节点，通常是 "Frame"。 |



### `CallEventMeta`



包含调用事件的元数据，提供有关其执行的详细信息。

| 字段         | 类型                   | 描述                                                         |
| ------------ | ---------------------- | ------------------------------------------------------------ |
| `ThreadId`   | integer                | 执行此事件的线程 ID。                                        |
| `ThreadName` | string (optional)      | 执行此事件的线程名称。                                       |
| `TimerId`    | integer                | 计时器的 ID。                                                |
| `TimerName`  | string (optional)      | 计时器的名称。对于此函数筛选的节点，通常是 "Frame"。         |
| `StartTime`  | float                  | 事件开始的时间戳，以秒为单位。                               |
| `EndTime`    | float                  | 事件结束的时间戳，以秒为单位。                               |
| `Duration`   | float                  | 事件的持续时间 (EndTime - StartTime)，以秒为单位。           |
| `CallDepth`  | integer                | 事件在调用堆栈中的深度。                                     |
| `event_id`   | integer                | 事件在整个追踪文件中的唯一标识符。                           |
| `call_stack` | List[`CallStackFrame`] | 一个调用堆栈帧列表，显示该节点的完整调用堆栈（不包括事件本身），顺序从上到下（调用者 -> 被调用者），每个元素包含一个堆栈帧的事件 ID 和计时器名称。 |



### `CallEventNode`



一个 `anytree` 节点，它包含一个 `CallEventMeta` 实例来存储元数据。它提供了调用事件的层次化表示。



### `ExceptionFrame`



表示已超过定义的性能阈值的帧，指示潜在的性能问题。

| 字段         | 类型              | 描述                                               |
| ------------ | ----------------- | -------------------------------------------------- |
| `event_id`   | integer           | 事件在整个追踪文件中的唯一标识符。                 |
| `ThreadName` | string (optional) | 执行此事件的线程名称。                             |
| `Duration`   | float             | 事件的持续时间 (EndTime - StartTime)，以秒为单位。 |
| `StartTime`  | float             | 事件开始的时间戳，以秒为单位。                     |
| `EndTime`    | float             | 事件结束的时间戳，以秒为单位。                     |



### `TimerMeta`



代表来自 Unreal Insights 'TimerStat.csv' 报告中单个计时器的元数据。所有时间单位均为秒 (s)。

| 字段               | 类型    | 描述                                                         |
| ------------------ | ------- | ------------------------------------------------------------ |
| `TimerName`        | string  | 计时器的名称，通常是函数名或作用域事件。                     |
| `count`            | integer | 在跟踪期间该计时器被触发的次数。                             |
| `inclusive_total`  | float   | 总包含时间，单位为秒 (s)。这是在此计时器及其调用的所有子计时器中所花费时间的总和。 |
| `inclusive_min`    | float   | 单次调用的最小包含时间，单位为秒 (s)。                       |
| `inclusive_max`    | float   | 单次调用的最大包含时间，单位为秒 (s)。                       |
| `inclusive_avg`    | float   | 该计时器的平均包含时间，单位为秒 (s)。                       |
| `inclusive_median` | float   | 该计时器的中位数包含时间，单位为秒 (s)。                     |
| `exclusive_total`  | float   | 总独占时间，单位为秒 (s)。这是在此计时器自身花费的时间总和，不包括在其调用的任何子计时器中花费的时间。 |
| `exclusive_min`    | float   | 单次调用的最小独占时间，单位为秒 (s)。                       |
| `exclusive_max`    | float   | 单次调用的最大独占时间，单位为秒 (s)。                       |
| `exclusive_avg`    | float   | 该计时器的平均独占时间，单位为秒 (s)。                       |
| `exclusive_median` | float   | 该计时器的中位数独占时间，单位为秒 (s)。                     |



### `CostDistribution`



用于分析和统计调用栈中各个Timer的性能开销分布的容器。

| 字段                  | 类型                   | 描述                                                         |
| --------------------- | ---------------------- | ------------------------------------------------------------ |
| `cost_distribution`   | Dict[str, `TimerMeta`] | 以TimerName为键的性能开销分布字典，包含每个Timer的详细统计信息。 |
| `total_analysis_time` | float                  | 整个分析时间窗口的总时长（秒）。                             |
| `total_events_count`  | integer                | 分析时间窗口内的总事件数量。                                 |

------



## API 端点 (工具)



这些是 MCP 服务器公开的函数。



### `get_exception_frame`
从Unreal Insights跟踪数据中识别并检索异常帧。异常帧定义为超过指定持续时间阈值的"Frame"事件，通常表示性能瓶颈或异常长的帧时间，可能导致应用程序的帧率下降或卡顿。
- **标签**: `performance`, `frame_analysis`, `exception_detection`
**参数:**
| 参数             | 类型   | 描述                                                         |
| ---------------- | ------ | ------------------------------------------------------------ |
| `utrace_file`    | string | 要分析的`.utrace`文件的路径或唯一标识符。                    |
| `max_frame_cost` | float  | 定义异常帧的最小持续时间阈值（以秒为单位）。持续时间 >= 此值的帧将被视为异常。 |
**返回:**
- **类型**: List[`ExceptionFrame`]
- **描述**: 表示超过指定持续时间阈值的帧的`ExceptionFrame`对象列表。每个帧包含`event_id`、`ThreadName`、`Duration`、`StartTime`和`EndTime`信息，用于进一步分析和优化。
### `getKeyNodes`
自动识别并检索调用栈指定子树中的关键性能事件。此工具使用智能过滤来发现对整体执行时间有重大贡献的节点，帮助开发者专注于最具影响力的优化机会。分析基于相对于子树根的时间成本比率，使其对分层性能分析非常有效。
- **标签**: `performance`, `optimization`, `call_tree`, `hotspot_detection`
**参数:**
| 参数            | 类型    | 默认值 | 描述                                                         |
| --------------- | ------- | ------ | ------------------------------------------------------------ |
| `utrace_file`   | string  |        | 要分析的`.utrace`文件的路径或唯一标识符。                    |
| `event_id`      | integer |        | 要查询的子树的根节点事件ID。这作为关键节点发现分析的起始点。 |
| `max_threshold` | float   | 0.005  | 用于发现关键节点的时间成本比率阈值（相对于子树根节点的时间成本）。更高的阈值导致更宽松的过滤条件，返回更多关键节点但可能包括不太关键的节点。 |
**返回:**
- **类型**: List[`CallEventMeta`]
- **描述**: 表示在子树中发现的关键节点的`CallEventMeta`对象列表。每个对象包含全面的元数据，包括时间信息、调用深度、线程详细信息和完整的调用栈信息。这些节点代表应优先进行优化的最关键性能事件。
### `getNodeMetaInfo`
检索由其`event_id`标识的特定事件节点的综合元数据信息。这包括详细的时间信息、线程上下文、调用栈层次结构和执行详细信息。对于特定性能事件的深入分析以及理解它们在更广泛调用树中的执行上下文至关重要。
- **标签**: `metadata`, `event_analysis`, `call_stack`
**参数:**
| 参数          | 类型    | 描述                                                         |
| ------------- | ------- | ------------------------------------------------------------ |
| `utrace_file` | string  | 要分析的`.utrace`文件的路径或唯一标识符。                    |
| `event_id`    | integer | 要查询的事件节点的唯一ID。此ID在跟踪数据中唯一标识特定事件。 |
**返回:**
- **类型**: `CallEventMeta`
- **描述**: 包含指定事件节点综合元数据的`CallEventMeta`对象。这包括时间信息（`StartTime`、`EndTime`、`Duration`）、线程详细信息（`ThreadId`、`ThreadName`）、计时器信息（`TimerId`、`TimerName`）、调用深度以及导致此事件的完整调用栈层次结构。
### `getCostDistribution`
计算并检索指定子树的综合性能成本分布分析。此分析提供子树内所有计时器的详细统计信息，包括包含/独占时间统计、调用频率和性能比率。基于Unreal Insights分析概念，它提供类似于`TimerStat.csv`报告的见解，但专注于特定的调用树分支。
- **标签**: `performance`, `statistics`, `cost_analysis`, `timer_distribution`
**参数:**
| 参数          | 类型    | 描述                                                         |
| ------------- | ------- | ------------------------------------------------------------ |
| `utrace_file` | string  | 要分析的`.utrace`文件的路径或唯一标识符。                    |
| `event_id`    | integer | 用作成本分布分析根节点的事件节点的唯一ID。分析将包括此节点及其在调用树中的所有后代。 |
**返回:**
- **类型**: `CostDistribution`
- **描述**: 包含指定子树综合性能分析结果的`CostDistribution`对象。这包括`TimerMeta`对象字典（按计时器名称键控），每个计时器的详细统计信息包括计数、包含/独占时间数据（总计、最小、最大、平均、中位数）、总分析时间和总事件计数。数据可用于识别性能热点、理解时间分布并优先考虑优化工作。