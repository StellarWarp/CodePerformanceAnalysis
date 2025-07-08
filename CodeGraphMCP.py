from enum import Enum
from typing import Annotated

from fastmcp import FastMCP


class LangType(str, Enum):
    CPP = "cpp"
    Lua = "lua"
    CSharp = "csharp"

# 通过环境变量传入lsp的地址

class LspMcpServer:
    def __init__(self,ip:str,port:int):
        self.mcp = FastMCP("lsp_mcp_server_exmple")
        self.ip = ip
        self.port = port

        @self.mcp.tool(description='''
                       跳转到定义,支持变量，函数，类，结构体，枚举，需要传入文件路径，行号，列号
                       ''')
        def go_to_defination(lang_type: Annotated[LangType, "要查询的编程语言的类型，支持三种输入 cpp, lua, csharp"],
                             file_path: Annotated[str, "要查询的文件的路径"],
                             line_number: Annotated[int, "要查询的行号"],
                             column_number: Annotated[int, "要查询的列号"]):
            pass

        @self.mcp.tool(description='''
                       查找引用,支持变量，函数，类，结构体，枚举，需要传入文件路径，行号，列号
                       ''')
        def find_references(lang_type: Annotated[LangType, "要查询的编程语言的类型，支持三种输入 cpp, lua, csharp"],
                             file_path: Annotated[str, "要查询的文件的路径"],
                             line_number: Annotated[int, "要查询的行号"],
                             column_number: Annotated[int, "要查询的列号"]):
            pass

        @self.mcp.tool(description='''
                       查找函数的定义,需要传入文件路径，行号，函数名
                       ''')
        def find_function_by_name_and_line(lang_type: Annotated[LangType, "要查询的编程语言的类型，支持三种输入 cpp, lua, csharp"],
                                           file_path: Annotated[str, "要查询的文件的路径"],
                                           line_number: Annotated[int, "要查询的行号"],
                                           function_name: Annotated[str, "要查询的函数名"]):
            pass

    def run(self):
        import uvicorn
        app = self.mcp.http_app()
        uvicorn.run(
            app,
            host=self.ip,      # ← 监听地址
            port=self.port,           # ← 监听端口
            log_level="info"
        )


# --- 服务器启动入口 ---
if __name__ == "__main__":
    SERVER_IP = "127.0.0.1"
    SERVER_PORT = 8000

    # 2. 检查所需的依赖库是否已安装
    try:
        import fastmcp
        import uvicorn
    except ImportError:
        print("错误：缺少必要的依赖库。")
        print("请通过以下命令安装: pip install fastmcp uvicorn")
        exit(1)

    # 3. 创建服务器实例
    lsp_server = LspMcpServer(ip=SERVER_IP, port=SERVER_PORT)

    # 4. 启动服务器
    lsp_server.run()
