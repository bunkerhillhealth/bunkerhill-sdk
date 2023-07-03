# How to generate gRPC stubs

Run the following command to generate code for Protocol Buffer messages and gRPC stubs:

```shell
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. bunkerhill/proto/inference.proto
```

Visit [grpc.io/](https://grpc.io/) to learn more about gRPC.
