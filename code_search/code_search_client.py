import grpc
import code_search_pb2
import code_search_pb2_grpc

def find_symbols(symbol_name):
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = code_search_pb2_grpc.SymbolSearchServiceStub(channel)
        
        request = code_search_pb2.SymbolRequest(symbol_name=symbol_name)
        
        try:
            response = stub.FindSymbols(request)
            
            for symbol in response.symbols:
                print(f"Symbol: {symbol.name}, Type: {symbol.type}, File: {symbol.file_path}, Line: {symbol.line_number}")
            
            return response.symbols
        except grpc.RpcError as e:
            print(f"gRPC error: {e.details()}")
            return []
        

def find_text(text):
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = code_search_pb2_grpc.SymbolSearchServiceStub(channel)
        request = code_search_pb2.TextSearchRequest(
            text=text,
            context_after=5,
            context_before=5,
            search_path = "",
            file_extension = "*.h;*.cpp"
            )
        
        try:
            response = stub.FindText(request)

            for result in response.results:
                print(f"File: {result.file_path}, Line: {result.line_number}, \nContext: \n{result.context}")
            
            return response.results
        except grpc.RpcError as e:
            print(f"gRPC error: {e.details()}")
            return []
        


if __name__ == '__main__':
    # 示例：查找符号 "MyClass"
    symbol_name = "func"
    find_text(symbol_name)