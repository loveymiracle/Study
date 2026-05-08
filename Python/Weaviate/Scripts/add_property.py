import weaviate
from weaviate.util import generate_uuid5
from datetime import datetime, timezone

client = weaviate.connect_to_custom(
    http_host="192.168.0.72",
    http_port=8081,
    http_secure=False,
    grpc_host="192.168.0.72",
    grpc_port=50052,
    grpc_secure=False,
)

try:
    collection = client.collections.get("ModelInfo")
    update_data = [
        {"model_name": "text-embedding-3-small", "tokenizer": "cl100k_base"},
        {"model_name": "multilingual-e5-base", "tokenizer": "intfloat"},
        {"model_name": "qwen3-embedding-0.6b", "tokenizer": "Qwen"},
    ]

    for data in update_data:
        target_uuid = generate_uuid5(data["model_name"])

        collection.data.update(uuid=target_uuid, properties=data)
        print(f"✅ 모델 '{data['model_name']}' 업데이트 완료 (UUID: {target_uuid})")

    print("\n🚀 모든 데이터 업데이트가 완료되었습니다.")

except Exception as e:
    print(f"🚨 업데이트 중 오류 발생: {e}")

finally:
    client.close()
