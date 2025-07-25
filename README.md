# README

- 环境

  ```bat
  conda env create -f env.yaml
  ```

- 修改Unreal Insights.exe位置

  ```python
  # flameGraphMCP.py
  mcp = FastMCP(name="Call Tree MCP Server")
  SESSION_BUFFER = dataloader.TraceDataManager(r'C:\Program Files\Epic Games\UE_5.6\Engine\Binaries\Win64\UnrealInsights.exe') <--
  ```

- 运行flameGraphMCP

  ```bat
  python flameGraphMCP.py
  ```

- 测试

  ```python
  # Test
  import asyncio
  
  from fastmcp import Client
  
  
  client = Client({
          "Unreal insight Call Tree": {
              'url': "http://localhost:8001/sse/",
              "transport": "sse"
          }
      })
  
  async def test_get_exception_frame(utrace_file: str,max_frame_cost:float):
      async with client:
          result = await client.call_tool("get_exception_frame", {"utrace_file": utrace_file, "max_frame_cost": max_frame_cost})
          print(result[0].text)
  
          # result = await client.call_tool("getKeyNodes", {"utrace_file": utrace_file, "event_id": 71405})
          # print(result[0].text)
  
          # result = await client.call_tool("getNodeMetaInfo", {"utrace_file": utrace_file, "event_id": 71405})
          # print(result[0].text)
  
          # result = await client.call_tool("getCostDistribution", {"utrace_file": utrace_file, "event_id": 71405})
          # print(result[0].text)
  
  if __name__ == '__main__':
      utrace_file = "C:\\\\Users\\\\lyq\\\\Desktop\\\\Work\\\\CodePerformanceAnalysis\\\\data\\\\CSV\\\\Test\\\\20250626_215834.utrace"
      asyncio.run(test_get_exception_frame(utrace_file,0.01666))
  ```

  